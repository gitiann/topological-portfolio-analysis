"""Tests for the topological-risk QP (stub)."""

from __future__ import annotations

import numpy as np
import pytest

from toporisk.portfolio import min_topological_risk_portfolio


@pytest.mark.xfail(raises=NotImplementedError, reason="TODO: portfolio QP", strict=True)
def test_weights_are_a_simplex_point():
    Q = np.diag([1.0, 2.0, 3.0])
    w = min_topological_risk_portfolio(Q)
    assert w.shape == (3,)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-9).all()


@pytest.mark.xfail(raises=NotImplementedError, reason="TODO: portfolio QP", strict=True)
def test_lower_risk_gets_more_weight():
    # diagonal Q -> closed form w_i proportional to 1/Lambda_i
    Q = np.diag([1.0, 4.0])
    w = min_topological_risk_portfolio(Q)
    assert np.allclose(w, [0.8, 0.2], atol=1e-3)
