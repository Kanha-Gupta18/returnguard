from __future__ import annotations

import json

import numpy as np
import pytest

import api.inference as inference
from api.schemas import ModelFeatureInput
from explainability.explainer import (
    _prepare_transaction_row,
    _vectorized_contributions,
    explain_transaction,
)


def _sample_input() -> ModelFeatureInput:
    return ModelFeatureInput(
        TransactionAmt_raw=250,
        TransactionAmt_log1p=np.log1p(250),
        hour_of_day=22,
        day_of_week=6,
        is_high_value=0,
        txn_deviation_from_customer_avg=100,
        V_V29=0,
        V_V37=0,
    )


def test_vectorized_perturbations_match_sequential_predictions() -> None:
    inference.load_runtime(force=True)
    model, feature_cols, reference_values = inference.get_explanation_context()
    row = _prepare_transaction_row(
        _sample_input().model_dump(),
        feature_cols,
    )
    reference_array = np.array(
        [reference_values[name] for name in feature_cols],
        dtype=float,
    )

    original, vectorized = _vectorized_contributions(
        model,
        row,
        reference_array,
    )
    sequential = []
    for index, reference_value in enumerate(reference_array):
        perturbed = row.copy()
        perturbed[0, index] = reference_value
        probability = float(model.predict_proba(perturbed)[:, 1][0])
        sequential.append(original - probability)

    assert np.allclose(vectorized, sequential, rtol=0, atol=1e-12)


def test_factor_groups_match_contribution_direction() -> None:
    inference.load_runtime(force=True)
    result = explain_transaction(
        fraud_prob=inference.predict_fraud_probability(_sample_input()),
        transaction_data=_sample_input().model_dump(),
    )

    assert result["positive_contributors"]
    assert result["negative_contributors"]
    assert all(
        factor["contribution"] > 0
        for factor in result["positive_contributors"]
    )
    assert all(
        factor["contribution"] < 0
        for factor in result["negative_contributors"]
    )


def test_prediction_pipeline_returns_a_typed_contract() -> None:
    inference.load_runtime(force=True)
    result = inference.predict_transaction(_sample_input())

    assert 0 <= result["risk_score"] <= 1
    assert result["risk_band"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["transaction_amount"] == 250
    assert len(result["top_risk_factors"]) <= 5
    assert len(result["top_protective_factors"]) <= 5


def test_runtime_rejects_a_model_checksum_mismatch(tmp_path) -> None:
    metadata = json.loads(
        inference.settings.model_metadata_path.read_text(encoding="utf-8")
    )
    metadata["artifact_sha256"] = "0" * 64
    metadata_path = tmp_path / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="model_checksum_mismatch"):
        inference.load_runtime(
            model_metadata_path=metadata_path,
            force=True,
        )
