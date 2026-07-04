"""Tests for sub-windowing (implemented) and Takens embedding (stub)."""

from __future__ import annotations

import numpy as np
import pytest

from toporisk.embedding import sub_windows, takens_embedding

# --- sub_windows: implemented -> pass ----------------------------------------

def test_sub_windows_count_and_shape():
    series = np.arange(252, dtype=float)
    wins = list(sub_windows(series, sub_len=126, shift=21))
    assert len(wins) == (252 - 126) // 21 + 1  # == 7
    assert all(w.shape == (126,) for w in wins)


def test_sub_windows_start_positions():
    series = np.arange(252, dtype=float)
    wins = list(sub_windows(series, sub_len=126, shift=21))
    assert wins[0][0] == 0
    assert wins[1][0] == 21
    assert wins[2][0] == 42


def test_sub_windows_rejects_bad_len():
    with pytest.raises(ValueError):
        list(sub_windows(np.zeros(10), sub_len=11, shift=1))


# --- takens_embedding: stub -> xfail until you implement it ------------------

#@pytest.mark.xfail(raises=NotImplementedError, reason="TODO: takens_embedding", strict=True)
def test_takens_toy(toy_series):
    # A = [1, -2, 3, -1, 2], d=2, tau=1  ->  the delay vectors from the discussion
    out = takens_embedding(toy_series, d=2, tau=1)
    expected = np.array([[1.0, -2.0], [-2.0, 3.0], [3.0, -1.0], [-1.0, 2.0]])
    assert out.shape == (4, 2)
    assert np.allclose(out, expected)


#@pytest.mark.xfail(raises=NotImplementedError, reason="TODO: takens_embedding", strict=True)
def test_takens_paper_shape():
    series = np.arange(126, dtype=float)
    out = takens_embedding(series, d=3, tau=1)
    assert out.shape == (124, 3)  # 126 - (3-1)*1 = 124 points in R^3


#@pytest.mark.xfail(raises=NotImplementedError, reason="TODO: takens_embedding", strict=True)
def test_takens_too_short_raises():
    with pytest.raises(ValueError):
        takens_embedding(np.array([1.0, 2.0]), d=3, tau=1)  # needs >= 3 points
