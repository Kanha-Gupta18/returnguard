"""Validated runtime configuration for the ReturnGuard API."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Environment-backed settings with safe local defaults."""

    model_config = SettingsConfigDict(
        env_prefix="RETURNGUARD_",
        extra="ignore",
    )

    app_name: str = "ReturnGuard Fraud Detection API"
    app_version: str = "12.0.0"
    environment: str = "development"
    build_sha: str = "development"

    model_path: Path = REPO_ROOT / "models" / "xgboost.joblib"
    explanation_reference_path: Path = (
        REPO_ROOT / "models" / "explanation_reference.json"
    )
    model_metadata_path: Path = REPO_ROOT / "models" / "model_metadata.json"

    allowed_origins: str = (
        "http://127.0.0.1:5173,http://localhost:5173"
    )
    max_concurrent_predictions: int = Field(default=2, ge=1, le=16)
    prediction_queue_timeout_seconds: float = Field(
        default=0.25,
        gt=0,
        le=10,
    )
    rate_limit_per_minute: int = Field(default=30, ge=1, le=10_000)
    log_level: str = "INFO"

    @field_validator(
        "model_path",
        "explanation_reference_path",
        "model_metadata_path",
        mode="before",
    )
    @classmethod
    def resolve_repo_path(cls, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in logging.getLevelNamesMapping():
            raise ValueError("log_level must be a standard Python logging level")
        return normalized

    @property
    def allowed_origin_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]


settings = Settings()
