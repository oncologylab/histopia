"""Numerical and support checks for scientific evaluation metrics."""

import numpy as np
import pytest

from histopia.study._evaluation import _metrics


def test_constant_float32_predictions_have_undefined_correlation():
    observed = np.linspace(0, 1, 1001, dtype=np.float32)
    predicted = np.full(1001, 0.1, dtype=np.float32)
    result = _metrics(observed, predicted)
    assert result["pearson_r"] is None
    assert result["mae"] == pytest.approx(
        np.abs(observed.astype(float) - predicted.astype(float)).mean()
    )


def test_missing_support_must_be_filtered_explicitly():
    with pytest.raises(ValueError, match="supported finite"):
        _metrics(np.array([1.0, np.nan]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="aligned"):
        _metrics(np.array([1.0]), np.array([1.0, 2.0]))


def test_empty_and_perfect_metrics():
    assert _metrics([], []) == {"n": 0, "mae": None, "rmse": None, "pearson_r": None}
    assert _metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == {
        "n": 3,
        "mae": 0.0,
        "rmse": 0.0,
        "pearson_r": 1.0,
    }
