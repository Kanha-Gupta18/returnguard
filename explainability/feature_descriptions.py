"""Human-readable descriptions for XGBoost features.

Maps feature names to understandable descriptions for the explainability layer.
These are NOT ML model features; they are purely documentation/translation.

DO NOT claim causality. Use language such as "contributed to the model's risk score."
"""

from __future__ import annotations
from typing import Dict, Optional


# Core mapping of feature names to human descriptions.
# Keyed by the exact feature name as used in the model pipeline.
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    # Transaction amount features
    "TransactionAmt_raw": "Transaction amount in original currency units",
    "TransactionAmt_log1p": "Log-transformed transaction amount",
    "is_high_value": "Transaction amount exceeds USD 500 threshold",
    # Temporal features
    "hour_of_day": "Hour of the day when transaction occurred",
    "day_of_week": "Day of the week when transaction occurred",
    # Customer deviation feature (key engineered feature)
    "txn_deviation_from_customer_avg": (
        "Transaction amount differs from the customer's historical average"
    ),
    # Card features (pseudo-anonymized)
    "card1": "Card identifier 1",
    "card2": "Card identifier 2",
    "card3": "Card identifier 3",
    "card4": "Card identifier 4",
    "card5": "Card identifier 5",
    "card6": "Card identifier 6",
    # Distance features (pseudo-anonymized)
    "dist1": "Distance feature 1",
    "dist2": "Distance feature 2",
    # Email domain features (pseudo-anonymized)
    "P_emaildomain": "Primary email domain",
    "R_emaildomain": "Registrar email domain",
    # V-features (pseudo-anonymized, part of the IEEE-CIS feature set)
    "V_V12": "V-feature V12 (anonymized count/sum feature)",
    # Customer ID (identifier, not used as model feature but included in feature set)
    "customer_id": "Customer identifier",
}


def get_description(feature_name: str, default: Optional[str] = None) -> Optional[str]:
    """Return a human-readable description for a feature name.

    Parameters
    ----------
    feature_name : str
        The exact feature name as used in the model.
    default : Optional[str], optional
        Value to return if the feature is not found in the mapping.
        Defaults to None.

    Returns
    -------
    Optional[str]
        Human-readable description, or None/default if not found.
    """
    return FEATURE_DESCRIPTIONS.get(feature_name, default)


def describe_feature(feature_name: str) -> str:
    """Return a human-readable description, with a safe fallback.

    Parameters
    ----------
    feature_name : str
        The exact feature name as used in the model.

    Returns
    -------
    str
        Description string. If the feature is not in the mapping,
        returns a safe default: "Feature: {feature_name}".
    """
    desc = FEATURE_DESCRIPTIONS.get(feature_name)
    if desc is not None:
        return desc
    return f"Feature: {feature_name}"