"""Explainability layer for the ReturnGuard fraud detection model.

Provides SHAP-based individual transaction explanations using the trained
XGBoost model at models/xgboost.joblib.

Do NOT retrain. Do NOT claim causality. Explanations are derived from actual
SHAP values only.
"""

from __future__ import annotations

from explainability.explainer import explain_transaction, explain_with_descriptions  # noqa: F401
from explainability.feature_descriptions import describe_feature, FEATURE_DESCRIPTIONS  # noqa: F401