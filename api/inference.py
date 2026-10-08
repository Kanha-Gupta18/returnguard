"""Model loading and the complete ReturnGuard prediction pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from pydantic import BaseModel

from api.schemas import ExpectedCosts, ModelFeatureInput, RiskFactor
from api.settings import settings
from decision_engine.config import DEFAULT_COST_MATRIX, DEFAULT_RISK_THRESHOLDS
from decision_engine.decide import decide

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeBundle:
    artifact: dict[str, Any]
    model: Any
    feature_cols: list[str]
    reference_values: dict[str, float]
    metadata: dict[str, Any]


_bundle: RuntimeBundle | None = None
_load_error_code: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file_handle:
            value = json.load(file_handle)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label}_missing") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label}_invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label}_invalid")
    return value


def _build_runtime_bundle(
    model_path: Path,
    explanation_reference_path: Path,
    model_metadata_path: Path,
) -> RuntimeBundle:
    if not model_path.is_file():
        raise RuntimeError("model_artifact_missing")

    metadata = _read_json(model_metadata_path, "model_metadata")
    reference_payload = _read_json(
        explanation_reference_path,
        "explanation_reference",
    )
    artifact_sha256 = _sha256(model_path)

    if metadata.get("artifact_sha256") != artifact_sha256:
        raise RuntimeError("model_checksum_mismatch")
    if reference_payload.get("model_sha256") != artifact_sha256:
        raise RuntimeError("explanation_reference_model_mismatch")

    installed_xgboost = package_version("xgboost")
    if metadata.get("framework_version") != installed_xgboost:
        raise RuntimeError("model_framework_version_mismatch")

    try:
        artifact = joblib.load(model_path)
        model = artifact["model"]
        feature_cols = [
            name
            for name in artifact["feature_cols"]
            if name != "customer_id"
        ]
        reference_values = {
            name: float(value)
            for name, value in reference_payload["medians"].items()
        }
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise RuntimeError("model_artifact_invalid") from exc

    if len(feature_cols) != metadata.get("feature_count"):
        raise RuntimeError("model_feature_count_mismatch")
    if set(feature_cols) != set(reference_values):
        raise RuntimeError("explanation_reference_feature_mismatch")
    if getattr(model, "n_features_in_", len(feature_cols)) != len(feature_cols):
        raise RuntimeError("model_input_shape_mismatch")

    return RuntimeBundle(
        artifact=artifact,
        model=model,
        feature_cols=feature_cols,
        reference_values=reference_values,
        metadata={**metadata, "artifact_sha256": artifact_sha256},
    )


def load_runtime(
    *,
    model_path: Path | None = None,
    explanation_reference_path: Path | None = None,
    model_metadata_path: Path | None = None,
    force: bool = False,
) -> None:
    """Load and validate every artifact required for prediction."""
    global _bundle, _load_error_code

    if _bundle is not None and not force:
        return

    try:
        candidate = _build_runtime_bundle(
            model_path or settings.model_path,
            explanation_reference_path or settings.explanation_reference_path,
            model_metadata_path or settings.model_metadata_path,
        )
    except RuntimeError as exc:
        _bundle = None
        _load_error_code = str(exc)
        logger.exception("ReturnGuard runtime validation failed")
        raise

    _bundle = candidate
    _load_error_code = None
    logger.info(
        "ReturnGuard runtime loaded",
        extra={
            "model_id": candidate.metadata["model_id"],
            "feature_count": len(candidate.feature_cols),
        },
    )


# Preserve the established private loader name for existing callers.
def _load_artifact() -> None:
    load_runtime()


def reset_runtime() -> None:
    global _bundle, _load_error_code
    _bundle = None
    _load_error_code = None


def runtime_status() -> dict[str, Any]:
    if _bundle is None:
        return {
            "ready": False,
            "error_code": _load_error_code or "runtime_not_loaded",
        }
    return {
        "ready": True,
        "error_code": None,
        "model_id": _bundle.metadata["model_id"],
        "feature_count": len(_bundle.feature_cols),
    }


def runtime_metadata() -> dict[str, Any]:
    if _bundle is None:
        raise RuntimeError("runtime_not_ready")
    return dict(_bundle.metadata)


def get_explanation_context() -> tuple[Any, list[str], dict[str, float]]:
    if _bundle is None:
        load_runtime()
    if _bundle is None:  # pragma: no cover - defensive after load_runtime
        raise RuntimeError("runtime_not_ready")
    return (
        _bundle.model,
        list(_bundle.feature_cols),
        dict(_bundle.reference_values),
    )


def _require_bundle() -> RuntimeBundle:
    if _bundle is None:
        load_runtime()
    if _bundle is None:  # pragma: no cover - defensive after load_runtime
        raise RuntimeError("runtime_not_ready")
    return _bundle


def _transaction_input_to_row_dict(input_data: BaseModel) -> dict[str, float]:
    bundle = _require_bundle()
    if not isinstance(input_data, BaseModel):
        raise TypeError("input_data must be a Pydantic model")

    row: dict[str, float] = {}
    for name in bundle.feature_cols:
        value = getattr(input_data, name, None)
        row[name] = 0.0 if value is None else float(value)
    return row


def predict_fraud_probability(input_data: ModelFeatureInput) -> float:
    bundle = _require_bundle()
    row = _transaction_input_to_row_dict(input_data)
    row_array = np.array(
        [[row[name] for name in bundle.feature_cols]],
        dtype=float,
    )
    probability = float(bundle.model.predict_proba(row_array)[:, 1][0])
    return max(0.0, min(1.0, probability))


def run_decision_engine(
    fraud_prob: float,
    transaction_amount: float,
    cost_matrix: Any = None,
    risk_thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    return decide(
        fraud_prob=fraud_prob,
        transaction_amount=transaction_amount,
        cost_matrix=cost_matrix or DEFAULT_COST_MATRIX,
        risk_thresholds=risk_thresholds or DEFAULT_RISK_THRESHOLDS,
    )


def run_explainability(
    input_data: ModelFeatureInput,
    fraud_prob: float,
) -> dict[str, Any]:
    bundle = _require_bundle()
    row = _transaction_input_to_row_dict(input_data)

    from explainability.explainer import explain_with_descriptions

    return explain_with_descriptions(
        transaction_data=row,
        fraud_prob=fraud_prob,
        model=bundle.model,
        feature_cols=bundle.feature_cols,
        reference_values=bundle.reference_values,
    )


def _risk_factors(
    contributors: Sequence[Mapping[str, Any]],
) -> list[RiskFactor]:
    return [
        RiskFactor(
            feature=str(contributor["feature"]),
            contribution=abs(float(contributor["contribution"])),
        )
        for contributor in contributors[:5]
    ]


def predict_transaction(
    input_data: ModelFeatureInput,
    cost_matrix: Any = None,
    risk_thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    fraud_probability = predict_fraud_probability(input_data)
    transaction_amount = float(input_data.TransactionAmt_raw or 0.0)
    decision = run_decision_engine(
        fraud_prob=fraud_probability,
        transaction_amount=transaction_amount,
        cost_matrix=cost_matrix,
        risk_thresholds=risk_thresholds,
    )
    explanation = run_explainability(input_data, fraud_probability)

    expected_costs = ExpectedCosts(
        approve=decision["expected_loss_if_approved"],
        review=decision["expected_loss_if_reviewed"],
        escalate=decision["expected_loss_if_rejected"],
    )
    return {
        "risk_score": decision["risk_score"],
        "risk_band": decision["risk_band"],
        "recommended_action": decision["recommended_action"],
        "expected_costs": expected_costs,
        "top_risk_factors": _risk_factors(
            explanation.get("positive_contributors", [])
        ),
        "top_protective_factors": _risk_factors(
            explanation.get("negative_contributors", [])
        ),
        "transaction_amount": transaction_amount,
    }
