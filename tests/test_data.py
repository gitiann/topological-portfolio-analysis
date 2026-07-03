"""Tests for data / returns. Implemented code -> should pass."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from toporisk.data import asset_series, load_prices, simple_returns


def test_simple_returns_of_constant_prices_is_zero():
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0]},
                          index=pd.date_range("2020-01-01", periods=3, freq="B"))
    r = simple_returns(prices)
    assert np.allclose(r["A"].to_numpy(), 0.0)


def test_simple_returns_values():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]},
                          index=pd.date_range("2020-01-01", periods=3, freq="B"))
    r = simple_returns(prices)
    assert np.allclose(r["A"].to_numpy(), [0.10, -0.10])


def test_simple_returns_drops_first_row():
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]},
                          index=pd.date_range("2020-01-01", periods=3, freq="B"))
    assert len(simple_returns(prices)) == 2


def test_simple_returns_rejects_nonpositive():
    prices = pd.DataFrame({"A": [1.0, 0.0]},
                          index=pd.date_range("2020-01-01", periods=2, freq="B"))
    with pytest.raises(ValueError):
        simple_returns(prices)


def test_load_prices_roundtrip(tmp_path):
    csv = tmp_path / "p.csv"
    csv.write_text("date,A,B\n2020-01-01,10,20\n2020-01-02,11,19\n")
    df = load_prices(csv)
    assert list(df.columns) == ["A", "B"]
    assert df.index[0] == pd.Timestamp("2020-01-01")


def test_asset_series_is_1d(toy_returns):
    s = asset_series(toy_returns, "B")
    assert s.ndim == 1 and len(s) == len(toy_returns)
