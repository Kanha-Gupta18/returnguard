"""Cost model for expected loss calculation in the decision engine.

Computes expected cost of three actions:
  1. APPROVE:   expected cost = P(fraud) × false_positive_cost
  2. REVIEW:    expected cost = human_review_cost  (fixed, regardless of probability)
  3. REJECT:    expected cost = (1 - P(fraud)) × false_negative_cost

Note: "rejected" in our system means "flagged for review / escalation",
never a customer-facing decline. The "false negative cost" here captures
the opportunity cost of missing fraud when we do NOT approve.

All costs are from the configured CostMatrix; see config.py for assumptions.
"""

from __future__ import annotations

from typing import Tuple
from decision_engine.config import CostMatrix, RiskBand


def expected_cost_approve(
    fraud_prob: float, cost_matrix: CostMatrix
) -> float:
    """Expected cost if the transaction is APPROVED.

    If the transaction is actually fraudulent (prob = fraud_prob) and we approve,
    we incur the false positive cost (chargeback/lost goods).
    If it's legitimate, cost is 0.
    """
    if not 0.0 <= fraud_prob <= 1.0:
        raise ValueError("fraud_prob must be in [0, 1]")
    return fraud_prob * cost_matrix.false_positive_cost


def expected_cost_review(
    fraud_prob: float, cost_matrix: CostMatrix
) -> float:
    """Expected cost if the transaction is sent for HUMAN REVIEW.

    Fixed cost regardless of probability. The fraud_prob parameter is accepted
    for interface consistency but does not affect the fixed review cost.
    """
    _ = fraud_prob  # unused, included for interface consistency
    return cost_matrix.human_review_cost


def expected_cost_reject(
    fraud_prob: float, cost_matrix: CostMatrix
) -> float:
    """Expected cost if the transaction is REJECTED (escalated for review).

    In our defensive system, "reject" means escalate to human investigation.
    The cost is the false negative cost: we miss the fraud opportunity,
    measured as a multiple of the transaction amount.

    Actually, for our system: rejecting = sending for review/escalation.
    The "false negative" cost here captures the risk of NOT catching fraud
    when we ultimately reject/block. This is (1 - P(fraud)) × cost_mult × amt.
    But since we don't have the amount in this sub-routine, we return
    the probability-weighted component.
    """
    # The false-negative cost is incurred when we fail to catch fraud.
    # If we "reject" (escalate), we are more likely to catch fraud,
    # so the effective cost depends on the system design.
    # Here we model it as: cost of missing fraud = (1 - fraud_prob) × multiplier.
    # Since the decision engine has access to the transaction amount,
    # the full expected cost is computed in decision.py.
    _ = fraud_prob  # framework; full cost computed with amount in decision.py
    return 0.0  # placeholder; actual cost computed with amount


def compute_risk_band(
    fraud_prob: float,
    thresholds: dict[str, float] | None = None,
) -> RiskBand:
    """Map a fraud probability to a risk band.

    Uses configurable thresholds; defaults:
      - LOW:     prob < 0.20
      - MEDIUM:  0.20 ≤ prob < 0.50
      - HIGH:    prob ≥ 0.50
    """
    if not 0.0 <= fraud_prob <= 1.0:
        raise ValueError("fraud_prob must be in [0, 1]")

    if thresholds is None:
        thresholds = {
            "LOW": 0.2,
            "MEDIUM": 0.5,
            "HIGH": 1.0,
        }

    if fraud_prob < thresholds["LOW"]:
        return "LOW"
    elif fraud_prob < thresholds["MEDIUM"]:
        return "MEDIUM"
    else:
        return "HIGH"


def expected_loss_approve(
    fraud_prob: float, transaction_amount: float, cost_matrix: CostMatrix
) -> float:
    """Expected loss if the transaction is APPROVED.

    Components:
      - If fraud (prob = fraud_prob): lose false_positive_cost
      - If legitimate (prob = 1 - fraud_prob): lose 0
    """
    return expected_cost_approve(fraud_prob, cost_matrix)


def expected_loss_review(
    fraud_prob: float, transaction_amount: float, cost_matrix: CostMatrix
) -> float:
    """Expected loss if the transaction is sent for HUMAN REVIEW.

    Fixed cost regardless of probability or amount.
    """
    return expected_cost_review(fraud_prob, cost_matrix)


def expected_loss_reject(
    fraud_prob: float, transaction_amount: float, cost_matrix: CostMatrix
) -> float:
    """Expected loss if the transaction is REJECTED/ESCALATED.

    In our defensive system, rejecting means escalating to human review.
    The false negative cost is the risk of missing fraud, measured as
    a multiple of the transaction amount.

    Cost = (1 - fraud_prob) × false_negative_cost × transaction_amount
    This represents the expected loss from not catching fraud
    when we ultimately do not approve.
    """
    fn_cost = (1.0 - fraud_prob) * cost_matrix.false_negative_cost * transaction_amount
    return fn_cost