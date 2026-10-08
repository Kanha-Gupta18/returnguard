from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from api.schemas import PredictionRequest


def test_v1_request_computes_derived_features_server_side() -> None:
    request = PredictionRequest(
        transaction_amount=600,
        hour_of_day=14,
        day_of_week=3,
        customer_average_amount=250,
        advanced_features={"card1": 12, "V_V29": -2},
    )

    model_input = request.to_model_input()

    assert model_input.TransactionAmt_raw == 600
    assert model_input.TransactionAmt_log1p == pytest.approx(math.log1p(600))
    assert model_input.is_high_value == 1
    assert model_input.txn_deviation_from_customer_avg == 350
    assert model_input.card1 == 12
    assert model_input.V_V29 == -2


@pytest.mark.parametrize(
    "field,value",
    [
        ("transaction_amount", 0),
        ("hour_of_day", 24),
        ("day_of_week", 7),
        ("customer_average_amount", -1),
    ],
)
def test_v1_request_rejects_invalid_ranges(field: str, value: float) -> None:
    payload = {
        "transaction_amount": 100,
        "hour_of_day": 10,
        "day_of_week": 2,
        "customer_average_amount": 80,
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


def test_v1_request_rejects_unknown_advanced_features() -> None:
    with pytest.raises(ValidationError, match="Unknown advanced feature"):
        PredictionRequest(
            transaction_amount=100,
            hour_of_day=10,
            day_of_week=2,
            customer_average_amount=80,
            advanced_features={"customer_fraud_rate": 0.9},
        )
