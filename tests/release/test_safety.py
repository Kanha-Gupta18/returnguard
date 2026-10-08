from __future__ import annotations

import pytest

from decision_engine.decide import decide


@pytest.mark.parametrize("fraud_probability", [0.5, 0.7, 0.9, 1.0])
@pytest.mark.parametrize("transaction_amount", [0.01, 10, 250, 1_000_000])
def test_high_risk_transactions_are_never_approved(
    fraud_probability: float,
    transaction_amount: float,
) -> None:
    result = decide(fraud_probability, transaction_amount)

    assert result["risk_band"] == "HIGH"
    assert result["recommended_action"] != "APPROVE"
