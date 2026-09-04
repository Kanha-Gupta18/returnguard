"""Decision engine that converts XGBoost fraud probability + transaction amount
into an economically justified action using an explicit cost matrix.

Defensive design:
- Never autonomously reject/decline a customer.
- "ESCALATE" means route to human investigation.
- All costs are configurable via CostMatrix.
- Risk bands based on probability are configurable thresholds.

The engine compares expected costs of three actions:
  1. APPROVE   : expected loss = P(fraud) × false_positive_cost
  2. REVIEW    : expected loss = human_review_cost (fixed)
  3. ESCALATE  : expected loss = (1 - P(fraud)) × false_negative_cost × amount

Safety constraint: ESCALATE is the minimum recommended action when
fraud probability exceeds the HIGH risk band threshold. Even if APPROVE
has lower expected cost in pure math, we never approve a clearly high-risk
transaction without human review.
"""

from __future__ import annotations

from typing import Literal, Tuple

from decision_engine.config import CostMatrix, RiskBand, DEFAULT_RISK_THRESHOLDS, DEFAULT_COST_MATRIX
from decision_engine.cost_model import (
    expected_loss_approve,
    expected_loss_reject,
    expected_loss_review,
    compute_risk_band,
)


Action = Literal["APPROVE", "HUMAN_REVIEW", "ESCALATE"]

# Safety floor: never APPROVE when risk is HIGH unless explicitly overridden.
# This is a hard constraint; the cost comparison can suggest APPROVE,
# but the safety constraint overrides it to ESCALATE.
SAFETY_PREVENT_APPROVE_HIGH_RISK = True


def decide(
    fraud_prob: float,
    transaction_amount: float,
    cost_matrix: CostMatrix | None = None,
    risk_thresholds: dict[RiskBand, float] | None = None,
) -> dict:
    """Determine the recommended action for a transaction.

    Parameters
    ----------
    fraud_prob : float
        XGBoost predicted probability of fraud in [0, 1].
    transaction_amount : float
        Transaction amount in currency units (e.g., USD).
    cost_matrix : CostMatrix, optional
        Cost matrix configuration. Uses DEFAULT_COST_MATRIX if None.
    risk_thresholds : dict[RiskBand, float], optional
        Probability thresholds for LOW/MEDIUM/HIGH bands.
        Uses DEFAULT_RISK_THRESHOLDS if None.

    Returns
    -------
    dict with keys:
        - risk_score:       float (the raw fraud probability)
        - risk_band:        str   ("LOW" / "MEDIUM" / "HIGH")
        - expected_loss_if_approved: float
        - expected_loss_if_reviewed: float
        - expected_loss_if_rejected: float
        - recommended_action: str   ("APPROVE" / "HUMAN_REVIEW" / "ESCALATE")
        - decision_reason:    str
    """
    # --- Input validation ---
    if not 0.0 <= fraud_prob <= 1.0:
        raise ValueError("fraud_prob must be in [0, 1]")
    if transaction_amount <= 0:
        raise ValueError("transaction_amount must be positive")

    # --- Use provided or default configurations ---
    cm = cost_matrix if cost_matrix is not None else DEFAULT_COST_MATRIX
    thresholds = risk_thresholds if risk_thresholds is not None else DEFAULT_RISK_THRESHOLDS

    # --- Compute risk band ---
    risk_band = compute_risk_band(fraud_prob, thresholds)

    # --- Compute expected losses for each action ---
    loss_approve = expected_loss_approve(fraud_prob, transaction_amount, cm)
    loss_review = expected_loss_review(fraud_prob, transaction_amount, cm)
    loss_reject = expected_loss_reject(fraud_prob, transaction_amount, cm)

    # --- Cost-sensitive selection ---
    # Build a mapping of action → expected loss
    losses: dict[Action, float] = {
        "APPROVE": loss_approve,
        "HUMAN_REVIEW": loss_review,
        "ESCALATE": loss_reject,
    }

    # Select action with minimum expected loss
    best_action: Action = min(losses, key=losses.get)
    best_loss = losses[best_action]

    # --- Safety constraint ---
    # Never autonomously approve a HIGH-risk transaction.
    # If the cost-minimizing action is APPROVE but risk is HIGH,
    # override to ESCALATE (human investigation).
    if SAFETY_PREVENT_APPROVE_HIGH_RISK and risk_band == "HIGH" and best_action == "APPROVE":
        best_action = "ESCALATE"
        # Re-compute the loss for the overridden action for the reason string
        best_loss = losses["ESCALATE"]

    # --- Generate human-readable decision reason ---
    reason_parts = [
        f"fraud_prob={fraud_prob:.4f}",
        f"amount=${transaction_amount:.2f}",
        f"risk_band={risk_band}",
        f"costs: approve={loss_approve:.2f}, review={loss_review:.2f}, escalate={loss_reject:.2f}",
    ]
    if best_action == "APPROVE":
        # Only reach here if safety constraint didn't block it
        reason_parts.append(f"min_expected_loss=approve")
    elif best_action == "HUMAN_REVIEW":
        reason_parts.append(f"min_expected_loss=review")
    else:
        reason_parts.append(f"min_expected_loss=escalate")

    decision_reason = "; ".join(reason_parts)

    # --- Construct and return the result ---
    result = {
        "risk_score": fraud_prob,
        "risk_band": risk_band,
        "expected_loss_if_approved": round(loss_approve, 4),
        "expected_loss_if_reviewed": round(loss_review, 4),
        "expected_loss_if_rejected": round(loss_reject, 4),
        "recommended_action": best_action,
        "decision_reason": decision_reason,
    }

    return result