"""Small smoke check for imports and the explanation reference artifact."""

from __future__ import annotations

from explainability.explainer import describe_feature, explain_transaction
from explainability.feature_descriptions import FEATURE_DESCRIPTIONS


def main() -> None:
    transaction = {
        "TransactionAmt_raw": 150.0,
        "hour_of_day": 14,
        "is_high_value": 0,
        "txn_deviation_from_customer_avg": 200.0,
        "V_V12": 0.0,
    }
    result = explain_transaction(
        fraud_prob=0.35,
        transaction_data=transaction,
    )

    assert describe_feature("txn_deviation_from_customer_avg")
    assert "TransactionAmt_raw" in FEATURE_DESCRIPTIONS
    assert 0.0 <= result["risk_probability"] <= 1.0
    print("ReturnGuard import and explanation smoke check passed.")


if __name__ == "__main__":
    main()
