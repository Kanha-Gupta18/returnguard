"""Perturbation-based explanations for the ReturnGuard model.

The explainer is pure at runtime: the API owns artifact loading and passes the
model, feature order, and training-reference medians into these functions.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from explainability.feature_descriptions import describe_feature, get_description


def _resolve_context(
    model: Any | None,
    feature_cols: Sequence[str] | None,
    reference_values: Mapping[str, float] | None,
) -> tuple[Any, list[str], dict[str, float]]:
    if model is not None and feature_cols is not None and reference_values is not None:
        return model, list(feature_cols), dict(reference_values)

    from api.inference import get_explanation_context

    return get_explanation_context()


def _prepare_transaction_row(
    transaction_data: dict[str, Any] | pd.DataFrame,
    feature_cols: Sequence[str],
) -> np.ndarray:
    if isinstance(transaction_data, dict):
        row_dict = {name: transaction_data.get(name, 0.0) for name in feature_cols}
        row_df = pd.DataFrame([row_dict])
    elif isinstance(transaction_data, pd.DataFrame):
        row_df = pd.DataFrame(
            {name: transaction_data.get(name, 0.0) for name in feature_cols},
            index=[0],
        )
    else:
        raise TypeError("transaction_data must be a dict or pandas DataFrame")

    for name in feature_cols:
        row_df[name] = pd.to_numeric(row_df[name], errors="coerce")
    return row_df[list(feature_cols)].fillna(0.0).to_numpy(dtype=float)


def _vectorized_contributions(
    model: Any,
    row_array: np.ndarray,
    reference_array: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Predict the original and every one-feature perturbation in one batch."""
    feature_count = row_array.shape[1]
    batch = np.repeat(row_array, feature_count + 1, axis=0)
    feature_indexes = np.arange(feature_count)
    batch[feature_indexes + 1, feature_indexes] = reference_array

    probabilities = model.predict_proba(batch)[:, 1].astype(float)
    original_probability = float(probabilities[0])

    # Contribution is the original score minus the reference-perturbed score.
    # Positive values therefore mean the observed feature increased risk.
    contributions = original_probability - probabilities[1:]
    return original_probability, contributions


def explain_transaction(
    fraud_prob: float,
    transaction_data: dict[str, Any] | pd.DataFrame,
    *,
    model: Any | None = None,
    feature_cols: Sequence[str] | None = None,
    reference_values: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if not 0.0 <= fraud_prob <= 1.0:
        raise ValueError("fraud_prob must be in [0, 1]")

    resolved_model, resolved_cols, resolved_refs = _resolve_context(
        model,
        feature_cols,
        reference_values,
    )
    row_array = _prepare_transaction_row(transaction_data, resolved_cols)
    reference_array = np.array(
        [resolved_refs[name] for name in resolved_cols],
        dtype=float,
    )
    original_probability, contribution_values = _vectorized_contributions(
        resolved_model,
        row_array,
        reference_array,
    )

    contributions: list[dict[str, Any]] = []
    for index, feature_name in enumerate(resolved_cols):
        contribution = float(contribution_values[index])
        contributions.append(
            {
                "feature": feature_name,
                "original_value": float(row_array[0, index]),
                "reference_value": float(reference_array[index]),
                "contribution": contribution,
                "direction": (
                    "INCREASED_RISK" if contribution > 0 else "DECREASED_RISK"
                ),
                "description": get_description(feature_name),
            }
        )

    positive_contributors = sorted(
        (item for item in contributions if item["contribution"] > 0),
        key=lambda item: item["contribution"],
        reverse=True,
    )[:5]
    negative_contributors = sorted(
        (item for item in contributions if item["contribution"] < 0),
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )[:5]

    total_positive = sum(item["contribution"] for item in positive_contributors)
    total_negative = sum(abs(item["contribution"]) for item in negative_contributors)
    if total_positive > total_negative:
        overall_direction = "INCREASED_RISK"
    elif total_negative > total_positive:
        overall_direction = "DECREASED_RISK"
    else:
        overall_direction = "UNKNOWN"

    ranked = sorted(
        contributions,
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )
    top = ranked[0] if ranked else None

    return {
        "risk_probability": original_probability,
        "feature_name": top["feature"] if top else "",
        "feature_value": top["original_value"] if top else None,
        "shap_contribution": abs(top["contribution"]) if top else 0.0,
        "direction": overall_direction,
        "positive_contributors": positive_contributors,
        "negative_contributors": negative_contributors,
    }


def explain_with_descriptions(
    transaction_data: dict[str, Any] | pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    result = explain_transaction(transaction_data=transaction_data, **kwargs)
    for contributor in (
        result["positive_contributors"] + result["negative_contributors"]
    ):
        contributor["description"] = (
            describe_feature(contributor["feature"])
            or contributor.get("description", "")
        )
    return result
