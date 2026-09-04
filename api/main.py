"""ReturnGuard FastAPI application.

Exposes two endpoints:
  GET  /health      -> health check
  POST /predict     -> fraud prediction + decision + explanation
"""

from __future__ import annotations

import api.inference as inf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import TransactionInput, PredictionResponse
from api.inference import predict_transaction

# Create FastAPI app
app = FastAPI(
    title="ReturnGuard Fraud Detection API",
    description="Fraud detection baseline using XGBoost + cost-sensitive decision engine",
    version="11.0.0",
)

# Add basic CORS so a local frontend can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Startup: load the XGBoost model artifact once.
# ------------------------------------------------------------------


@app.on_event("startup")
def _startup_load_artifact() -> None:
    inf._load_artifact()


# ------------------------------------------------------------------
# GET /health
# ------------------------------------------------------------------


@app.get(
    "/health",
    tags=["health"],
    summary="Health check",
    description="Return service status.",
)
async def health() -> JSONResponse:
    """Health check endpoint.

    Returns:
        JSONResponse with {"status": "ok"}
    """
    return JSONResponse(content={"status": "ok"})


# ------------------------------------------------------------------
# POST /predict
# ------------------------------------------------------------------


@app.post(
    "/predict",
    tags=["prediction"],
    summary="Predict fraud and recommend action",
    description="""
    Run a transaction through the full ReturnGuard pipeline.

    1. XGBoost model predicts fraud probability
    2. Decision engine computes risk band and recommended action
       (never autonomously approves HIGH-risk)
    3. Perturbation-based explainability identifies top risk
       and protective factors
    4. Unified prediction response returned

    Input features are all optional; missing values default to 0.0.
    The model artifact is loaded once at startup, not per request.

    Safety constraint: if the cost-minimizing action is APPROVE but the
    risk band is HIGH, the action is overridden to ESCALATE.
    """,
)
async def predict(
    transaction: TransactionInput,
) -> PredictionResponse:
    """Predict fraud and recommend action for a transaction.

    Parameters
    ----------
    transaction : TransactionInput
        Transaction features (all optional; missing → 0.0).

    Returns
    -------
    PredictionResponse
        Unified prediction response.

    Raises
    ------
    HTTPException
        If inference fails (stack traces are never exposed to the client).
    """
    try:
        result = predict_transaction(transaction)
        # Pydantic v2 model validator ensures recommended_action is valid
        return result
    except ValueError as e:
        # Client input or business logic violation — 400
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Model not loaded, preprocessing missing, etc. — 503
        raise HTTPException(status_code=503, detail="Service unavailable")
    except Exception as e:  # pragma: no cover
        # Unexpected error — 500, never expose stack traces
        raise HTTPException(status_code=500, detail="Internal prediction error")


# ------------------------------------------------------------------
# Entrypoint (when run directly: `python api/main.py`)
# ------------------------------------------------------------------


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)