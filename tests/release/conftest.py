from __future__ import annotations

import pytest

import api.inference as inference


@pytest.fixture(autouse=True)
def reset_prediction_runtime():
    inference.reset_runtime()
    yield
    inference.reset_runtime()
