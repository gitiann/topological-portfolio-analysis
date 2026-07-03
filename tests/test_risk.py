"""Tests for risk assembly. risk_matrix is implemented; Lambda_i is a stub."""

from __future__ import annotations

import numpy as np
import pytest

from toporisk.risk import asset_topological_risk, risk_matrix

# --- risk_matrix: implemented -> pass ----------------------------------------

def test_risk_matrix_is_diagonal():
    Q = risk_matrix(np.array([1.0, 4.0, 9.0]))
    assert np.allclose(Q, np.diag([1.0, 4.0, 9.0]))


def test_risk_matrix_rejects_negative():
    with pytest.raises(ValueError):
        risk_matrix(np.array([1.0, -2.0]))


# --- asset_topological_risk: stub -> xfail -----------------------------------

@pytest.mark.xfail(raises=NotImplementedError, reason="TODO: asset_topological_risk", strict=True)
def test_asset_topological_risk_nonnegative():
    rng = np.random.default_rng(0)
    series = rng.normal(scale=0.01, size=252)
    val = asset_topological_risk(series, sub_len=126, shift=21, d=3, tau=1)
    assert val >= 0.0
