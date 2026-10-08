"""FastAPI application factory and HTTP contract for ReturnGuard."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import api.inference as inference
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.limits import SlidingWindowRateLimiter
from api.schemas import (
    ErrorResponse,
    HealthResponse,
    MetadataResponse,
    ModelFeatureInput,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
)
from api.settings import Settings, settings

logger = logging.getLogger("returnguard.api")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_PREDICTION_PATHS = {"/predict", "/api/v1/predict"}


class JsonFormatter(logging.Formatter):
    """Compact JSON logs suitable for local output and hosted log drains."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in (
            "request_id",
            "method",
            "path",
            "status_code",
            "duration_ms",
            "model_id",
            "feature_count",
        ):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = record.exc_info[0].__name__
        return json.dumps(payload, separators=(",", ":"))


def _configure_logging(level: str) -> None:
    for logger_name in ("returnguard.api", "api.inference"):
        application_logger = logging.getLogger(logger_name)
        application_logger.setLevel(level)
        application_logger.propagate = False
        if not any(
            isinstance(handler.formatter, JsonFormatter)
            for handler in application_logger.handlers
        ):
            handler = logging.StreamHandler()
            handler.setFormatter(JsonFormatter())
            application_logger.addHandler(handler)


def _request_id(request: Request) -> str:
    supplied = request.headers.get("X-Request-ID", "")
    if _REQUEST_ID_PATTERN.fullmatch(supplied):
        return supplied
    return uuid.uuid4().hex


def _error_payload(
    request: Request,
    code: str,
    message: str,
    *,
    fields: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
            "fields": fields,
        }
    }


def create_app(config: Settings | None = None) -> FastAPI:
    app_settings = config or settings
    _configure_logging(app_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        inference.reset_runtime()
        try:
            inference.load_runtime(
                model_path=app_settings.model_path,
                explanation_reference_path=(
                    app_settings.explanation_reference_path
                ),
                model_metadata_path=app_settings.model_metadata_path,
                force=True,
            )
        except RuntimeError:
            logger.exception("API started without a ready prediction runtime")

        app.state.prediction_semaphore = asyncio.Semaphore(
            app_settings.max_concurrent_predictions
        )
        app.state.rate_limiter = SlidingWindowRateLimiter(
            app_settings.rate_limit_per_minute
        )
        yield
        inference.reset_runtime()

    app = FastAPI(
        title=app_settings.app_name,
        description=(
            "XGBoost risk scoring with cost-sensitive decisioning and "
            "perturbation-based explanations."
        ),
        version=app_settings.app_version,
        lifespan=lifespan,
    )
    app.state.settings = app_settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.allowed_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = _request_id(request)
        started = time.perf_counter()

        if request.url.path in _PREDICTION_PATHS:
            client_key = request.client.host if request.client else "unknown"
            allowed, retry_after = await app.state.rate_limiter.check(client_key)
            if not allowed:
                response = JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content=_error_payload(
                        request,
                        "rate_limit_exceeded",
                        "Prediction request limit exceeded. Try again shortly.",
                    ),
                    headers={"Retry-After": str(retry_after)},
                )
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Process-Time-Ms"] = str(duration_ms)
        logger.info(
            "request_complete",
            extra={
                "request_id": request.state.request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        fields = [
            {
                "field": ".".join(str(part) for part in error["loc"] if part != "body"),
                "message": error["msg"],
                "type": error["type"],
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                request,
                "validation_error",
                "The request contains invalid fields.",
                fields=fields,
            ),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        code_by_status = {
            400: "invalid_request",
            404: "not_found",
            429: "rate_limit_exceeded",
            503: "service_unavailable",
        }
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                request,
                code_by_status.get(exc.status_code, "request_failed"),
                message,
            ),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.exception(
            "Unhandled API error",
            extra={"request_id": getattr(request.state, "request_id", None)},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_payload(
                request,
                "internal_error",
                "An unexpected error occurred.",
            ),
        )

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    @app.get("/health/live", response_model=HealthResponse, tags=["health"])
    async def liveness() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get(
        "/health/ready",
        response_model=ReadinessResponse,
        responses={503: {"model": ReadinessResponse}},
        tags=["health"],
    )
    async def readiness() -> ReadinessResponse | JSONResponse:
        runtime = inference.runtime_status()
        response = ReadinessResponse(
            status="ready" if runtime["ready"] else "not_ready",
            model_id=runtime.get("model_id"),
            feature_count=runtime.get("feature_count"),
        )
        if not runtime["ready"]:
            return JSONResponse(status_code=503, content=response.model_dump())
        return response

    @app.get(
        "/api/v1/meta",
        response_model=MetadataResponse,
        responses={503: {"model": ErrorResponse}},
        tags=["metadata"],
    )
    async def metadata() -> MetadataResponse:
        if not inference.runtime_status()["ready"]:
            raise HTTPException(
                status_code=503,
                detail="Prediction runtime is not ready.",
            )
        model_metadata = inference.runtime_metadata()
        return MetadataResponse(
            api_version=app_settings.app_version,
            build_sha=app_settings.build_sha,
            model_id=model_metadata["model_id"],
            model_sha256=model_metadata["artifact_sha256"],
            model_framework=model_metadata["framework"],
            model_framework_version=model_metadata["framework_version"],
            feature_contract_version=model_metadata["feature_contract_version"],
            feature_count=model_metadata["feature_count"],
            decision_policy_version=model_metadata["decision_policy_version"],
        )

    async def execute_prediction(
        model_input: ModelFeatureInput,
    ) -> PredictionResponse:
        if not inference.runtime_status()["ready"]:
            raise HTTPException(
                status_code=503,
                detail="Prediction runtime is not ready.",
            )

        try:
            await asyncio.wait_for(
                app.state.prediction_semaphore.acquire(),
                timeout=app_settings.prediction_queue_timeout_seconds,
            )
        except TimeoutError as exc:
            raise HTTPException(
                status_code=503,
                detail="Prediction capacity is busy. Try again shortly.",
            ) from exc

        try:
            result = await run_in_threadpool(
                inference.predict_transaction,
                model_input,
            )
            return PredictionResponse.model_validate(result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail="Prediction runtime is not ready.",
            ) from exc
        finally:
            app.state.prediction_semaphore.release()

    @app.post(
        "/api/v1/predict",
        response_model=PredictionResponse,
        responses={
            422: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["prediction"],
    )
    async def predict_v1(payload: PredictionRequest) -> PredictionResponse:
        return await execute_prediction(payload.to_model_input())

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        deprecated=True,
        tags=["compatibility"],
    )
    async def predict_legacy(payload: ModelFeatureInput) -> PredictionResponse:
        return await execute_prediction(payload)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
