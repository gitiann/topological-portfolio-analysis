"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_series() -> np.ndarray:
    """The 5-point series used in the design discussion: A = [1, -2, 3, -1, 2]."""
    return np.array([1.0, -2.0, 3.0, -1.0, 2.0])


@pytest.fixture
def two_clusters() -> np.ndarray:
    """Two tight, well-separated clusters in R^2 (8 points total)."""
    rng = np.random.default_rng(0)
    a = rng.normal(loc=[0.0, 0.0], scale=0.01, size=(4, 2))
    b = rng.normal(loc=[10.0, 0.0], scale=0.01, size=(4, 2))
    return np.vstack([a, b])


@pytest.fixture
def toy_returns() -> pd.DataFrame:
    """A small positive-price-derived returns panel (300 days, 3 assets)."""
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    scales = [0.005, 0.012, 0.025]
    data = np.column_stack([rng.normal(scale=s, size=300) for s in scales])
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C"])
