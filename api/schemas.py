"""Pydantic schemas for the ReturnGuard API request/response."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


# ------------------------------------------------------------------
# Transaction input: one field per model feature (48 features, excluding customer_id)
# ------------------------------------------------------------------

class TransactionInput(BaseModel):
    """Input transaction features for the XGBoost model.

    All fields are optional; missing values default to 0.0
    at inference time. Column names must match the trained model's
    feature column names exactly.
    """

    # Transaction amount features
    TransactionAmt_raw: Optional[float] = None
    TransactionAmt_log1p: Optional[float] = None

    # Time features
    hour_of_day: Optional[float] = None
    day_of_week: Optional[float] = None

    # Derivative features
    is_high_value: Optional[float] = None
    txn_deviation_from_customer_avg: Optional[float] = None

    # Card features (6)
    card1: Optional[float] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[float] = None
    card5: Optional[float] = None
    card6: Optional[float] = None

    # Address features (2)
    addr1: Optional[float] = None
    addr2: Optional[float] = None

    # Distance features (2)
    dist1: Optional[float] = None
    dist2: Optional[float] = None

    # Email domain features (2) - cat.coded integers
    P_emaildomain: Optional[float] = None
    R_emaildomain: Optional[float] = None

    # V features (V_V12 through V_V41 = 30 features)
    V_V12: Optional[float] = None
    V_V13: Optional[float] = None
    V_V14: Optional[float] = None
    V_V15: Optional[float] = None
    V_V16: Optional[float] = None
    V_V17: Optional[float] = None
    V_V18: Optional[float] = None
    V_V19: Optional[float] = None
    V_V20: Optional[float] = None
    V_V21: Optional[float] = None
    V_V22: Optional[float] = None
    V_V23: Optional[float] = None
    V_V24: Optional[float] = None
    V_V25: Optional[float] = None
    V_V26: Optional[float] = None
    V_V27: Optional[float] = None
    V_V28: Optional[float] = None
    V_V29: Optional[float] = None
    V_V30: Optional[float] = None
    V_V31: Optional[float] = None
    V_V32: Optional[float] = None
    V_V33: Optional[float] = None
    V_V34: Optional[float] = None
    V_V35: Optional[float] = None
    V_V36: Optional[float] = None
    V_V37: Optional[float] = None
    V_V38: Optional[float] = None
    V_V39: Optional[float] = None
    V_V40: Optional[float] = None
    V_V41: Optional[float] = None

    @validator(
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "dist1",
        "dist2",
        "P_emaildomain",
        "R_emaildomain",
    )
    def features_must_be_numeric(cls, v):
        if v is not None and not isinstance(v, (int, float)):
            raise ValueError("Feature must be numeric")
        return v


# ------------------------------------------------------------------
# Risk factor entries for top_risk_factors / top_protective_factors
# ------------------------------------------------------------------

class RiskFactor(BaseModel):
    """A contributing factor identified by feature name and contribution."""

    feature: str = Field(..., description="Feature name")
    contribution: float = Field(
        ..., description="Absolute contribution to probability change"
    )


# ------------------------------------------------------------------
# Expected costs per action
# ------------------------------------------------------------------

class ExpectedCosts(BaseModel):
    """Expected cost for each possible action."""

    approve: float = Field(..., ge=0, description="Expected loss if approved")
    review: float = Field(..., ge=0, description="Expected loss if reviewed")
    escalate: float = Field(..., ge=0, description="Expected loss if escalated")


# ------------------------------------------------------------------
# Complete prediction response
# ------------------------------------------------------------------

class PredictionResponse(BaseModel):
    """Unified response from the /predict endpoint."""

    risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Predicted fraud probability"
    )
    risk_band: str = Field(
        ..., description="Risk band: LOW | MEDIUM | HIGH"
    )
    recommended_action: str = Field(
        ...,
        description="APPROVE | HUMAN_REVIEW | ESCALATE",
    )
    expected_costs: ExpectedCosts = Field(
        ..., description="Expected costs per action"
    )
    top_risk_factors: List[RiskFactor] = Field(
        default_factory=list,
        description="Top features increasing fraud probability",
    )
    top_protective_factors: List[RiskFactor] = Field(
        default_factory=list,
        description="Top features decreasing fraud probability",
    )
    transaction_amount: float = Field(
        ..., ge=0.0, description="Transaction amount in currency units"
    )

    @validator("recommended_action")
    def action_must_be_valid(cls, v):
        if v not in {"APPROVE", "HUMAN_REVIEW", "ESCALATE"}:
            raise ValueError(
                "recommended_action must be one of: APPROVE, HUMAN_REVIEW, ESCALATE"
            )
        return v