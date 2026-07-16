"""toporisk — topological risk portfolios.

Reproduction skeleton for Goel, Sharma & Kanniainen, "Class of topological
portfolios: Are they better than classical portfolios?" (2026), arXiv:2601.03974.

Pipeline (per training window)
------------------------------
For each asset i, independently:
  1. simple returns of asset i                                    (data.py)
  2. split the return series into overlapping sub-windows         (embedding.py)
     (length T~=126 days, shift h=21)
  3. Takens delay-embed each sub-window: d=3, tau=1  -> 124 x 3   (embedding.py)
  4. Vietoris-Rips H0 persistence diagram of the cloud            (topology.py)
  5. persistence landscape (k=1) of the H0 diagram                (landscape.py)
  6. L^1 norm of each sub-window landscape, and of the mean       (landscape.py)
     landscape across sub-windows
  7. asset risk  Lambda_i = sum_j (||eta^(j)|| - ||mean eta||)^2  (risk.py)
Then across assets:
  8. Q = diag(Lambda_i)                                           (risk.py)
  9. w* = argmin w^T Q w  s.t.  sum w = 1, w >= 0                 (portfolio.py)

"""

from __future__ import annotations

__version__ = "0.1.0"

from toporisk.data import load_prices, simple_returns
from toporisk.embedding import sub_windows, takens_embedding
from toporisk.topology import finite_pairs, persistence_diagram_h0

__all__ = [
    "__version__",
    "load_prices",
    "simple_returns",
    "sub_windows",
    "takens_embedding",
    "persistence_diagram_h0",
    "finite_pairs",
]
