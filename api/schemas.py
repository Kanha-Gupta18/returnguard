"""Pydantic v2 request and response schemas for ReturnGuard."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RiskBand = Literal["LOW", "MEDIUM", "HIGH"]
RecommendedAction = Literal["APPROVE", "HUMAN_REVIEW", "ESCALATE"]

ADVANCED_FEATURE_NAMES = frozenset(
    [
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
    ]
    + [f"V_V{index}" for index in range(12, 42)]
)


class ModelFeatureInput(BaseModel):
    """Legacy model-aligned input used by the compatibility endpoint."""

    model_config = ConfigDict(extra="forbid")

    TransactionAmt_raw: float | None = None
    TransactionAmt_log1p: float | None = None
    hour_of_day: float | None = None
    day_of_week: float | None = None
    is_high_value: float | None = None
    txn_deviation_from_customer_avg: float | None = None

    card1: float | None = None
    card2: float | None = None
    card3: float | None = None
    card4: float | None = None
    card5: float | None = None
    card6: float | None = None
    addr1: float | None = None
    addr2: float | None = None
    dist1: float | None = None
    dist2: float | None = None
    P_emaildomain: float | None = None
    R_emaildomain: float | None = None

    V_V12: float | None = None
    V_V13: float | None = None
    V_V14: float | None = None
    V_V15: float | None = None
    V_V16: float | None = None
    V_V17: float | None = None
    V_V18: float | None = None
    V_V19: float | None = None
    V_V20: float | None = None
    V_V21: float | None = None
    V_V22: float | None = None
    V_V23: float | None = None
    V_V24: float | None = None
    V_V25: float | None = None
    V_V26: float | None = None
    V_V27: float | None = None
    V_V28: float | None = None
    V_V29: float | None = None
    V_V30: float | None = None
    V_V31: float | None = None
    V_V32: float | None = None
    V_V33: float | None = None
    V_V34: float | None = None
    V_V35: float | None = None
    V_V36: float | None = None
    V_V37: float | None = None
    V_V38: float | None = None
    V_V39: float | None = None
    V_V40: float | None = None
    V_V41: float | None = None

    @field_validator("*", mode="after")
    @classmethod
    def values_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Feature values must be finite numbers")
        return value


# Preserve the established import name for internal callers and compatibility.
TransactionInput = ModelFeatureInput


class PredictionRequest(BaseModel):
    """Small public input contract for the versioned API."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "transaction_amount": 250.0,
                "hour_of_day": 14,
                "day_of_week": 3,
                "customer_average_amount": 150.0,
                "advanced_features": {"V_V29": 0.0, "V_V37": 0.0},
            }
        },
    )

    transaction_amount: float = Field(gt=0, le=10_000_000)
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    customer_average_amount: float = Field(ge=0, le=10_000_000)
    advanced_features: dict[str, float] = Field(default_factory=dict)

    @field_validator("transaction_amount", "customer_average_amount")
    @classmethod
    def amounts_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Amounts must be finite numbers")
        return value

    @field_validator("advanced_features")
    @classmethod
    def validate_advanced_features(
        cls,
        value: dict[str, float],
    ) -> dict[str, float]:
        unknown = sorted(set(value) - ADVANCED_FEATURE_NAMES)
        if unknown:
            raise ValueError(
                f"Unknown advanced feature(s): {', '.join(unknown)}"
            )
        invalid = [
            name
            for name, feature_value in value.items()
            if not math.isfinite(feature_value)
            or abs(feature_value) > 1_000_000_000
        ]
        if invalid:
            raise ValueError(
                "Advanced feature values are out of range: "
                f"{', '.join(sorted(invalid))}"
            )
        return value

    def to_model_input(self) -> ModelFeatureInput:
        amount = self.transaction_amount
        return ModelFeatureInput(
            TransactionAmt_raw=amount,
            TransactionAmt_log1p=math.log1p(amount),
            hour_of_day=float(self.hour_of_day),
            day_of_week=float(self.day_of_week),
            is_high_value=1.0 if amount > 500 else 0.0,
            txn_deviation_from_customer_avg=(
                amount - self.customer_average_amount
            ),
            **self.advanced_features,
        )


class RiskFactor(BaseModel):
    feature: str
    contribution: float = Field(ge=0)


class ExpectedCosts(BaseModel):
    approve: float = Field(ge=0)
    review: float = Field(ge=0)
    escalate: float = Field(ge=0)


class PredictionResponse(BaseModel):
    risk_score: float = Field(ge=0, le=1)
    risk_band: RiskBand
    recommended_action: RecommendedAction
    expected_costs: ExpectedCosts
    top_risk_factors: list[RiskFactor] = Field(default_factory=list)
    top_protective_factors: list[RiskFactor] = Field(default_factory=list)
    transaction_amount: float = Field(ge=0)


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_id: str | None = None
    feature_count: int | None = None


class MetadataResponse(BaseModel):
    api_version: str
    build_sha: str
    model_id: str
    model_sha256: str
    model_framework: str
    model_framework_version: str
    feature_contract_version: str
    feature_count: int
    decision_policy_version: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str
    fields: list[dict[str, str]] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
