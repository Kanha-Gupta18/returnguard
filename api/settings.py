"""Runtime paths shared by the API and inference helpers."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_path_from_env(name: str, default: str) -> Path:
    """Resolve an optional environment path from the repository root."""
    configured = Path(os.getenv(name, default)).expanduser()
    if not configured.is_absolute():
        configured = REPO_ROOT / configured
    return configured.resolve()


MODEL_PATH = _repo_path_from_env(
    "RETURNGUARD_MODEL_PATH",
    "models/xgboost.joblib",
)
EXPLANATION_REFERENCE_PATH = _repo_path_from_env(
    "RETURNGUARD_EXPLANATION_REFERENCE_PATH",
    "models/explanation_reference.json",
)
