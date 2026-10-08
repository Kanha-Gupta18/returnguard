from __future__ import annotations

import asyncio
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from api.main import create_app
from api.settings import Settings


async def _client_for(app):
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    )


def test_health_metadata_and_v1_prediction() -> None:
    async def scenario() -> None:
        app = create_app(
            Settings(build_sha="test-build", rate_limit_per_minute=100)
        )
        async with app.router.lifespan_context(app):
            async with await _client_for(app) as client:
                live = await client.get(
                    "/health/live",
                    headers={"X-Request-ID": "test-live"},
                )
                ready = await client.get("/health/ready")
                metadata = await client.get("/api/v1/meta")
                prediction = await client.post(
                    "/api/v1/predict",
                    json={
                        "transaction_amount": 250,
                        "hour_of_day": 22,
                        "day_of_week": 6,
                        "customer_average_amount": 150,
                        "advanced_features": {"V_V29": 0, "V_V37": 0},
                    },
                )

        assert live.status_code == 200
        assert live.json() == {"status": "ok"}
        assert live.headers["X-Request-ID"] == "test-live"
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"
        assert metadata.status_code == 200
        assert metadata.json()["build_sha"] == "test-build"
        assert metadata.json()["feature_count"] == 48
        assert prediction.status_code == 200
        assert prediction.json()["risk_band"] in {"LOW", "MEDIUM", "HIGH"}
        assert prediction.json()["recommended_action"] in {
            "APPROVE",
            "HUMAN_REVIEW",
            "ESCALATE",
        }
        assert prediction.headers["X-Request-ID"]

    asyncio.run(scenario())


def test_legacy_prediction_route_remains_compatible() -> None:
    async def scenario() -> None:
        app = create_app(Settings(rate_limit_per_minute=100))
        async with app.router.lifespan_context(app):
            async with await _client_for(app) as client:
                response = await client.post(
                    "/predict",
                    json={
                        "TransactionAmt_raw": 250,
                        "hour_of_day": 22,
                        "day_of_week": 6,
                        "is_high_value": 0,
                        "txn_deviation_from_customer_avg": 100,
                    },
                )

        assert response.status_code == 200
        assert response.json()["transaction_amount"] == 250

    asyncio.run(scenario())


def test_validation_errors_use_the_stable_error_contract() -> None:
    async def scenario() -> None:
        app = create_app(Settings(rate_limit_per_minute=100))
        async with app.router.lifespan_context(app):
            async with await _client_for(app) as client:
                response = await client.post(
                    "/api/v1/predict",
                    json={
                        "transaction_amount": -5,
                        "hour_of_day": 99,
                        "day_of_week": 1,
                        "customer_average_amount": 0,
                        "unexpected": "not accepted",
                    },
                )

        body = response.json()
        assert response.status_code == 422
        assert body["error"]["code"] == "validation_error"
        assert body["error"]["request_id"] == response.headers["X-Request-ID"]
        assert {field["field"] for field in body["error"]["fields"]} == {
            "transaction_amount",
            "hour_of_day",
            "unexpected",
        }

    asyncio.run(scenario())


def test_liveness_stays_up_while_readiness_reports_missing_model(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        app = create_app(
            Settings(
                model_path=tmp_path / "missing-model.joblib",
                rate_limit_per_minute=100,
            )
        )
        async with app.router.lifespan_context(app):
            async with await _client_for(app) as client:
                live = await client.get("/health/live")
                ready = await client.get("/health/ready")
                metadata = await client.get("/api/v1/meta")

        assert live.status_code == 200
        assert ready.status_code == 503
        assert ready.json() == {
            "status": "not_ready",
            "model_id": None,
            "feature_count": None,
        }
        assert metadata.status_code == 503
        assert metadata.json()["error"]["code"] == "service_unavailable"

    asyncio.run(scenario())
