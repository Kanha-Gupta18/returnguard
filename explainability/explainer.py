"""Explainability layer for the ReturnGuard fraud detection model.

Provides transaction-level explanations using perturbation-based feature attribution
via the trained XGBoost model at models/xgboost.joblib.

Does NOT retrain anything.
Does NOT claim causality.
Explanations are derived from model prediction changes, not SHAP values.

The public API consists of:
- explain_transaction(fraud_prob, transaction_data) -> explanation dict
- explain_with_descriptions(transaction_data) -> enriched explanation dict
- describe_feature(feature_name) -> human-readable description
"""

from __future__ import annotations

import joblib
import numpy as np
import os
import sys
import pandas as pd
from typing import Any, Dict, List, Optional

from explainability.feature_descriptions import get_description, describe_feature

# Ensure repo root is on path so `models.*` imports work regardless of cwd
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Load the existing XGBoost model artifact
_ARTIFACT_PATH = "models/xgboost.joblib"
_artifact = joblib.load(_ARTIFACT_PATH)
_clf = _artifact["model"]

# Model feature columns (excluding customer_id which is an identifier)
_all_feature_cols = [c for c in _artifact["feature_cols"] if c != "customer_id"]
_N_FEATURES = len(_all_feature_cols)

# ------------------------------------------------------------------
# Background sample for reference values (small subset of TRAIN only)
# ------------------------------------------------------------------
# Read only the columns the model expects; coerce non-numeric to 0.0.
_train = pd.read_csv("data/splits/train.csv")

# Build background dataset column by column so we don't need full alignment.
_background_cols = []
for c in _all_feature_cols:
    if c in _train.columns:
        s = pd.to_numeric(_train[c], errors="coerce")
    else:
        s = pd.Series([0.0] * len(_train), dtype=float)
    _background_cols.append(s)

_background_df = pd.concat(_background_cols, axis=1)
# Guarantee the column order matches _all_feature_cols
_background_df = _background_df[_all_feature_cols]
# Fill any remaining NaN with 0.0 and convert to float numpy array
_background_data = _background_df.fillna(0.0).values.astype(float)

# Determine a neutral reference value for each feature from the background sample.
# For numeric features, use the median of the background sample (robust to outliers).
_background_medians = np.median(_background_data, axis=0)
# For binary/boolean-like features (median is 0 or 1), ensure we have a sensible reference.
_reference_values = {c: float(_background_medians[i]) for i, c in enumerate(_all_feature_cols)}


# ------------------------------------------------------------------
# Core explanation: perturbation-based feature contributions
# ------------------------------------------------------------------

def _prepare_transaction_row(transaction_data: dict | pd.DataFrame) -> np.ndarray:
    """Convert a transaction dict/DataFrame into a numeric row array.

    Only the features the model expects are kept; missing features default to 0.0.
    Non-numeric values are coerced to 0.0 via pd.to_numeric.
    """
    if isinstance(transaction_data, dict):
        row_dict = {c: transaction_data.get(c, 0.0) for c in _all_feature_cols}
        row_df = pd.DataFrame([row_dict])
    elif isinstance(transaction_data, pd.DataFrame):
        row_df = pd.DataFrame(
            {c: transaction_data.get(c, 0.0) for c in _all_feature_cols}, index=[0]
        )
    else:
        raise TypeError("transaction_data must be a dict or pd.DataFrame")

    # Coerce all values to numeric; non-numeric become NaN then 0.0
    for c in _all_feature_cols:
        row_df[c] = pd.to_numeric(row_df[c], errors="coerce")
    row_df = row_df.fillna(0.0)

    # Select only the columns the model expects, in the correct order, as a numpy array
    row_array = row_df[_all_feature_cols].values.astype(float)
    return row_array


def _model_predict(row_array: np.ndarray) -> float:
    """Run the XGBoost model and return the predicted fraud probability."""
    # The model's predict_proba returns shape (1, 2); take the probability of class 1.
    prob = float(_clf.predict_proba(row_array)[:, 1][0])
    return prob


def explain_transaction(
    fraud_prob: float,
    transaction_data: dict | pd.DataFrame,
    *,
    reference_values: Optional[Dict[str, float]] = None,
) -> dict:
    """Generate a perturbation-based explanation for a single transaction.

    For each model feature, the feature is replaced with a reference value
    (the median of the background training sample, or a user-supplied dict).
    The model is re-predicted, and the change in fraud probability is recorded.

    Features are then ranked by the absolute change in probability.

    Parameters
    ----------
    fraud_prob : float
        The XGBoost predicted probability of fraud in [0, 1].
        Provided for context; the method recomputes the prediction from the
        supplied transaction_data.
    transaction_data : dict or pd.DataFrame
        Feature values for the transaction. Keys should match the model's
        expected feature names. Unknown keys are ignored; missing keys default
        to the reference value.
    reference_values : dict, optional
        Mapping of feature name -> reference value. If None, the method uses
        the median of the background training sample for each feature.

    Returns
    -------
    dict with keys:
        - risk_probability : float
          The model's predicted fraud probability for the transaction.
        - feature_name : str
          The feature with the largest absolute contribution.
        - feature_value : float
          The original value of that feature in the transaction.
        - shap_contribution : float
          The absolute change in probability caused by perturbing that feature.
        - direction : str
          "INCREASED_RISK" if the perturbation increased the model's fraud
          probability, "DECREASED_RISK" if it decreased it.
        - positive_contributors : list[dict]
          Top features whose perturbation increased the fraud probability.
          Each dict has keys: feature, original_value, reference_value,
          contribution, direction, description.
        - negative_contributors : list[dict]
          Top features whose perturbation decreased the fraud probability.
          Same schema as positive_contributors.
    """
    # ------------------------------------------------------------------
    # Validate inputs.
    # ------------------------------------------------------------------
    if not (0.0 <= fraud_prob <= 1.0):
        raise ValueError("fraud_prob must be in [0, 1]")

    # 1. Prepare the transaction row and compute the original prediction.
    # ------------------------------------------------------------------
    row_array = _prepare_transaction_row(transaction_data)
    original_prob = _model_predict(row_array)

    # ------------------------------------------------------------------
    # 2. Perturb each feature and record the change in probability.
    # ------------------------------------------------------------------
    contributions: List[Dict[str, Any]] = []

    for idx, feature_name in enumerate(_all_feature_cols):
        # Original value from the transaction
        original_value = float(row_array[0, idx])

        # Reference value to perturb to (median of background training sample,
        # or user-supplied dict; falls back to module-level _reference_values).
        ref_val = _reference_values.get(feature_name, 0.0) if reference_values is None else reference_values.get(feature_name, 0.0)

        # Build a copy of the row where this single feature is replaced.
        perturbed_row = row_array.copy()
        perturbed_row[0, idx] = ref_val

        perturbed_prob = _model_predict(perturbed_row)
        contribution = perturbed_prob - original_prob  # positive => risk increased

        # Human-readable description
        description = get_description(feature_name)

        contributions.append(
            {
                "feature": feature_name,
                "original_value": original_value,
                "reference_value": ref_val,
                "contribution": contribution,
                "direction": "INCREASED_RISK" if contribution > 0 else "DECREASED_RISK",
                "description": description,
            }
        )

    # ------------------------------------------------------------------
    # 3. Rank by absolute contribution and split into positive/negative.
    # ------------------------------------------------------------------
    # Sort by absolute contribution descending.
    sorted_by_abs = sorted(contributions, key=lambda x: abs(x["contribution"]), reverse=True)

    # Top features whose perturbation increased risk (positive contribution).
    positive_contributors = sorted_by_abs[:5]  # top up to 5
    # Sort positive contributors by contribution descending (most risk-increasing first).
    positive_contributors.sort(key=lambda x: x["contribution"], reverse=True)

    # Top features whose perturbation decreased risk (negative contribution).
    negative_contributors = sorted_by_abs[5:10]  # next up to 5
    # Sort negative contributors by contribution descending (closest to 0 first).
    # For negative values: -0.05 > -0.15, so descending puts closest-to-0 first.
    negative_contributors.sort(key=lambda x: x["contribution"], reverse=True)

    # ------------------------------------------------------------------
    # 4. Determine overall direction based on summed contributions.
    # ------------------------------------------------------------------
    total_pos = sum(c["contribution"] for c in positive_contributors)
    total_neg = sum(abs(c["contribution"]) for c in negative_contributors)

    if total_pos > total_neg:
        overall_direction = "INCREASED_RISK"
    elif total_neg > total_pos:
        overall_direction = "DECREASED_RISK"
    else:
        overall_direction = "UNKNOWN"

    # ------------------------------------------------------------------
    # 5. Return the structured explanation.
    # ------------------------------------------------------------------
    result: Dict[str, Any] = {
        "risk_probability": float(fraud_prob),
        "feature_name": "",
        "feature_value": None,
        "shap_contribution": 0.0,
        "direction": overall_direction,
        "positive_contributors": [
            {
                "feature": c["feature"],
                "original_value": c["original_value"],
                "reference_value": c["reference_value"],
                "contribution": c["contribution"],
                "direction": c["direction"],
                "description": c["description"],
            }
            for c in positive_contributors
        ],
        "negative_contributors": [
            {
                "feature": c["feature"],
                "original_value": c["original_value"],
                "reference_value": c["reference_value"],
                "contribution": c["contribution"],
                "direction": c["direction"],
                "description": c["description"],
            }
            for c in negative_contributors
        ],
    }

    # Set the top single feature for convenience.
    if positive_contributors:
        result["feature_name"] = positive_contributors[0]["feature"]
        result["feature_value"] = positive_contributors[0]["original_value"]
        result["shap_contribution"] = positive_contributors[0]["contribution"]
    elif negative_contributors:
        result["feature_name"] = negative_contributors[0]["feature"]
        result["feature_value"] = negative_contributors[0]["original_value"]
        result["shap_contribution"] = negative_contributors[0]["contribution"]
    else:
        result["feature_name"] = ""
        result["feature_value"] = None
        result["shap_contribution"] = 0.0

    return result


def explain_with_descriptions(
    transaction_data: dict | pd.DataFrame,
    **kwargs: Any,
) -> dict:
    """Generate an explanation and enrich feature names with human descriptions.

    Parameters
    ----------
    transaction_data : dict or pd.DataFrame
        See explain_transaction() for parameter details.
    **kwargs : passed through to explain_transaction().

    Returns
    -------
    dict same structure as explain_transaction(), but with descriptions
    populated in each contributor dict.
    """
    result = explain_transaction(transaction_data=transaction_data, **kwargs)

    for c in result["positive_contributors"]:
        c["description"] = describe_feature(c["feature"]) or c.get("description", "")

    for c in result["negative_contributors"]:
        c["description"] = describe_feature(c["feature"]) or c.get("description", "")

    return result