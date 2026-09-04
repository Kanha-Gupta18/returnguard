"""Inference module for ReturnGuard.

Connects the existing trained XGBoost model, decision engine,
and perturbation-based explainability layer.

Loads the model artifact once (singleton/lazy) and does NOT retrain.
Does NOT load merged.csv or features.csv.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repo root is on path so `decision_engine` and `explainability`
# imports work regardless of the cwd when `api.main` is imported.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import decision engine components at module level so they're always available
from decision_engine.decide import decide as _decide_func
from decision_engine.cost_model import expected_loss_approve, expected_loss_review, expected_loss_reject
from decision_engine.config import DEFAULT_COST_MATRIX, DEFAULT_RISK_THRESHOLDS

import logging
from typing import Any, Dict, List, Optional

import numpy as np

# Logging setup
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Singleton / lazy loader for the XGBoost model artifact
# ------------------------------------------------------------------

_ARTIFACT_PATH = "models/xgboost.joblib"
_artifact: Optional[Any] = None
_clf: Optional[Any] = None
_feature_cols: Optional[List[str]] = None
_n_features: Optional[int] = None


def _load_artifact() -> None:
    """Load the XGBoost model artifact once (lazy initialization)."""
    global _artifact, _clf, _feature_cols, _n_features

    if _artifact is not None:
        return  # already loaded

    import joblib as _joblib

    try:
        _artifact = _joblib.load(_ARTIFACT_PATH)
        _clf = _artifact["model"]
        # Full feature column list as stored in the artifact (includes customer_id).
        # Remove customer_id so the column count matches TransactionInput (48 features).
        all_feature_cols = list(_artifact["feature_cols"])
        _feature_cols = [c for c in all_feature_cols if c != "customer_id"]
        _n_features = len(_feature_cols)
        logger.info(
            "XGBoost model artifact loaded: %d features (including customer_id)",
            _n_features,
        )
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to load model artifact from %s: %s", _ARTIFACT_PATH, exc)
        _clf = None
        _feature_cols = []
        _n_features = 0


# -----------------------------------------------------------------#
# Feature row preparation (uses the explainer's existing logic)
# # -----------------------------------------------------------------#


def _transaction_input_to_row_dict(
    input_data: "TransactionInput",
) -> dict:
    """Convert a TransactionInput Pydantic model to a feature dict.

    Missing features default to 0.0. Column ordering follows the model's
    expected feature names.

    This function mirrors the preparation logic in
    explainability.explainer._prepare_transaction_row so that the
    explainability module can work with the same feature space.
    """
    from pydantic import BaseModel

    # Ensure the model artifact is loaded
    _load_artifact()

    if _clf is None or _feature_cols is None:
        raise RuntimeError("Model artifact could not be loaded")

    # Build a dict with exactly the columns the model expects.
    # Pydantic fields that don't map to a model column are ignored;
    # missing model columns default to 0.0.
    row_dict: dict = {}
    for col in _feature_cols:
        # Get the value from the input; default to 0.0 if None/missing
        val = getattr(input_data, col, None)
        if val is None:
            val = 0.0
        # Ensure numeric
        try:
            row_dict[col] = float(val)
        except (TypeError, ValueError):
            row_dict[col] = 0.0

    return row_dict


# -----------------------------------------------------------------#
# Core prediction pipeline
# # -----------------------------------------------------------------#


def predict_fraud_probability(
    input_data: "TransactionInput",
) -> float:
    """Run the XGBoost model and return the fraud probability.

    Parameters
    ----------
    input_data : TransactionInput
        Transaction features (all optional; missing → 0.0).

    Returns
    -------
    float
        Predicted probability of fraud in [0, 1].
    """
    from api.schemas import TransactionInput as _TI

    if not isinstance(input_data, _TI):
        raise TypeError("input_data must be a TransactionInput instance")

    _load_artifact()

    if _clf is None:
        raise RuntimeError("Model artifact could not be loaded")

    # --- Prepare feature row ---
    row_dict = _transaction_input_to_row_dict(input_data)
    row_array = np.array([row_dict[_feature_cols[i]] for i in range(_n_features)]).reshape(1, -1)

    # The model's predict_proba returns shape (1, 2); class 1 = fraud
    prob = float(_clf.predict_proba(row_array)[:, 1][0])

    # Clip to [0, 1] for safety
    return max(0.0, min(1.0, prob))


def run_decision_engine(
    fraud_prob: float,
    transaction_amount: float,
    cost_matrix: Any = None,
    risk_thresholds: dict = None,
) -> dict:
    """Run the decision engine and return the decision result.

    Parameters
    ----------
    fraud_prob : float
        XGBoost predicted fraud probability in [0, 1].
    transaction_amount : float
        Transaction amount in currency units.
    cost_matrix : CostMatrix, optional
        Override cost matrix. Uses DEFAULT_COST_MATRIX if None.
    risk_thresholds : dict, optional
        Override risk band thresholds. Uses DEFAULT_RISK_THRESHOLDS if None.

    Returns
    -------
    dict
        Decision engine output dict with risk_score, risk_band,
        recommended_action, expected costs, etc.
    """
    from decision_engine.decide import decide
    from decision_engine.config import DEFAULT_COST_MATRIX, DEFAULT_RISK_THRESHOLDS

    if cost_matrix is None:
        cost_matrix = DEFAULT_COST_MATRIX
    if risk_thresholds is None:
        risk_thresholds = DEFAULT_RISK_THRESHOLDS

    result = decide(
        fraud_prob=fraud_prob,
        transaction_amount=transaction_amount,
        cost_matrix=cost_matrix,
        risk_thresholds=risk_thresholds,
    )

    return result


def run_explainability(
    input_data: "TransactionInput",
    fraud_prob: float,
) -> dict:
    """Run perturbation-based explainability and return contributor dicts.

    Parameters
    ----------
    input_data : TransactionInput
        Transaction features (all optional; missing → 0.0).
    fraud_prob : float
        Current fraud probability from the model.

    Returns
    -------
    dict
        Explanation dict with positive_contributors and negative_contributors,
        each a list of dicts with keys: feature, original_value,
        reference_value, contribution, direction, description.
    """
    from api.inference import _transaction_input_to_row_dict  # local import to avoid circular

    # Ensure model is loaded
    _load_artifact()

    if _clf is None:
        return {"positive_contributors": [], "negative_contributors": []}

    # Prepare the transaction row dict (same format used by the explainer)
    row_dict = _transaction_input_to_row_dict(input_data)

    from explainability.explainer import explain_with_descriptions

    try:
        expl_result = explain_with_descriptions(
            transaction_data=row_dict,
            fraud_prob=fraud_prob,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Explainability module failed: %s", exc)
        return {"positive_contributors": [], "negative_contributors": []}

    return expl_result


# -----------------------------------------------------------------#
# Unified prediction endpoint logic (FastAPI-independent)
# # -----------------------------------------------------------------#


def predict_transaction(
    input_data: "TransactionInput",
    cost_matrix: Any = None,
    risk_thresholds: dict = None,
) -> Dict[str, Any]:
    """Run a single transaction through the full ReturnGuard pipeline.

    Pipeline:
      1. XGBoost model → fraud probability
      2. Decision engine → risk band, recommended action, expected costs
      3. Explainability → top risk and protective factors
      4. Unified dict compatible with PredictionResponse

    Parameters
    ----------
    input_data : TransactionInput
        Transaction features (all optional; missing → 0.0).
    cost_matrix : CostMatrix, optional
        Override cost matrix. Uses DEFAULT_COST_MATRIX if None.
    risk_thresholds : dict, optional
        Override risk band thresholds. Uses DEFAULT_RISK_THRESHOLDS if None.

    Returns
    -------
    dict
        Unified prediction result with keys:
        - risk_score: float
        - risk_band: str ("LOW"|"MEDIUM"|"HIGH")
        - recommended_action: str ("APPROVE"|"HUMAN_REVIEW"|"ESCALATE")
        - expected_costs: dict with "approve", "review", "escalate"
        - top_risk_factors: list of {feature, contribution}
        - top_protective_factors: list of {feature, contribution}
        - transaction_amount: float
    """
    from api.schemas import ExpectedCosts, RiskFactor  # local import

    # Step 1: Fraud probability
    fraud_prob = predict_fraud_probability(input_data)

    # Step 2: Transaction amount from input (TransactionAmt_raw if provided,
    # otherwise derive from available features or 0.0)
    txn_amount = float(getattr(input_data, "TransactionAmt_raw", 0.0) or 0.0)

    # Step 3: Decision engine
    decision_result = run_decision_engine(
        fraud_prob=fraud_prob,
        transaction_amount=txn_amount,
        cost_matrix=cost_matrix,
        risk_thresholds=risk_thresholds,
    )

    # Step 4: Explanation
    expl_result = run_explainability(input_data, fraud_prob)

    # Map positive contributors (risk-increasing) → top_risk_factors
    positive_contributors = expl_result.get("positive_contributors", [])[:5]
    top_risk_factors: List[RiskFactor] = [
        RiskFactor(feature=c["feature"], contribution=abs(float(c["contribution"])))
        for c in positive_contributors
    ]

    # Map negative contributors (risk-decreasing) → top_protective_factors
    negative_contributors = expl_result.get("negative_contributors", [])[:5]
    top_protective_factors: List[RiskFactor] = [
        RiskFactor(feature=c["feature"], contribution=abs(float(c["contribution"])))
        for c in negative_contributors
    ]

    # Step 5: Construct unified response dict
    unified = {
        "risk_score": decision_result["risk_score"],
        "risk_band": decision_result["risk_band"],
        "recommended_action": decision_result["recommended_action"],
        "expected_costs": {
            "approve": decision_result["expected_loss_if_approved"],
            "review": decision_result["expected_loss_if_reviewed"],
            "escalate": decision_result["expected_loss_if_rejected"],
        },
        "top_risk_factors": top_risk_factors,
        "top_protective_factors": top_protective_factors,
        "transaction_amount": txn_amount,
    }

    return unified