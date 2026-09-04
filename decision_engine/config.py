"""Configuration module for the cost-sensitive decision engine.

All costs are ASSUMPTIONS and can be overridden by the user.
Do NOT claim these are Razorpay's actual business costs.
"""

from __future__ import annotations

from typing import Literal

#: Risk level bands for fraud probability.
#: Used throughout the decision engine to categorise risk and
#: configure probability thresholds.
#: Values are always one of: "LOW", "MEDIUM", "HIGH".
RiskBand = Literal["LOW", "MEDIUM", "HIGH"]

#: Default probability thresholds mapping risk bands to ranges.
#: Can be overridden via ``DecisionEngineConfig.risk_thresholds``.
DEFAULT_RISK_THRESHOLDS: dict[RiskBand, float] = {
    "LOW": 0.2,       # prob < 0.20  → LOW
    "MEDIUM": 0.5,    # 0.20 ≤ prob < 0.50 → MEDIUM
    "HIGH": 1.0,      # prob ≥ 0.50        → HIGH
}

#: Default action recommendations per risk band.
#: These are overridden by the cost-sensitive comparison in ``decide()``.
DEFAULT_BAND_ACTIONS: dict[RiskBand, str] = {
    "LOW": "APPROVE",
    "MEDIUM": "HUMAN_REVIEW",
    "HIGH": "ESCALATE",
}


class CostMatrix:
    """Cost matrix for decision evaluation.

    Costs are per-transaction and should be positive numbers.

    - false_positive_cost: Cost when a fraudulent transaction is approved
      (the "lost" goods/chargeback cost). Assumed 750.
    - false_negative_cost: Cost multiplier for not catching fraud.
      Multiplied by the transaction amount. Assumed 1.0×.
    - human_review_cost: Fixed cost of performing manual review.
      Assumed 100.

    All values are ASSUMPTIONS and not Razorpay's actual business costs.
    """

    def __init__(
        self,
        false_positive_cost: float = 750.0,
        false_negative_cost: float = 1.0,
        human_review_cost: float = 100.0,
    ):
        self.false_positive_cost = false_positive_cost
        self.false_negative_cost = false_negative_cost
        self.human_review_cost = human_review_cost

    def __repr__(self):
        return (
            f"CostMatrix(false_positive_cost={self.false_positive_cost}, "
            f"false_negative_cost={self.false_negative_cost}, "
            f"human_review_cost={self.human_review_cost})"
        )


#: Default cost matrix instance used by the decision engine when none is provided.
DEFAULT_COST_MATRIX = CostMatrix()