"""The topological-risk portfolio (TDA-PO).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def min_topological_risk_portfolio(Q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Long-only minimum-topological-risk weights.  [TODO]

    Solve the convex quadratic program

        minimize    w^T Q w
        subject to  sum_i w_i = 1,   w_i >= 0.

    Q = diag(Lambda_i) is diagonal and PSD, so this is convex. Two routes:

      - cvxpy: ``cvxpy.Problem(cvxpy.Minimize(cvxpy.quad_form(w, Q)), [...])``.
      - closed form: for a strictly-positive diagonal Q the KKT conditions give
        an explicit solution and the non-negativity constraint turns out to be
        inactive. Deriving it (and seeing *why* w >= 0 is automatically
        satisfied) is worth doing by hand before you code it -- it tells you what
        this "portfolio" actually is.

    Return a 1-D array of length n summing to 1 with all entries >= 0.

    Args:
        Q: ``(n, n)`` diagonal PSD topological-risk matrix.

    Returns:
        Optimal weight vector, shape ``(n,)``.
    """
    lambdas = np.diag(Q)
    w = np.zeros(len(lambdas))
    pos = lambdas > 0
    if not pos.any():
        raise ValueError("all topological risks are zero; no investable asset")
    inv = 1.0 / lambdas[pos]
    w[pos] = inv / inv.sum()
    return w
     
