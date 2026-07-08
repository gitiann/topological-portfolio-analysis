"""Asset topological risk and the risk matrix Q.

``asset_topological_risk`` is the capstone: it composes every earlier piece and
is where the norm-of-mean subtlety actually bites. It is a stub. ``risk_matrix``
is trivial glue.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from toporisk.embedding import (  # noqa: F401  (used when you implement)
    sub_windows,
    takens_embedding,
)
from toporisk.landscape import lp_norm, mean_landscape, persistence_landscape  # noqa: F401
from toporisk.topology import finite_pairs, persistence_diagram_h0  # noqa: F401


def asset_topological_risk(
    series: NDArray[np.float64],
    sub_len: int = 126,
    shift: int = 21,
    d: int = 3,
    tau: int = 1,
    p: float = 1.0,
    num_layers: int = 1,
    resolution: int = 500,
) -> float:
    global_max = 0.0
    landscape = []
    finite_diagrams = []
    for sub_window in sub_windows(series, sub_len, shift):
        point_cloud = takens_embedding(sub_window, d=d, tau=tau)
        diagram = persistence_diagram_h0(point_cloud)
        finite_diagrams.append(finite_pairs(diagram))
        for death in finite_diagram[:, 1]:
            if global_max < death:
                global_max = death

    for finite_diagram in finite_diagrams:
        landscape.append(persistence_landscape(finite_diagram, num_layers=num_layers, resolution=resolution, x_range=(0.0, global_max)))
    
    eta_bar = mean_landscape(landscape)
    Lambda = sum((lp_norm(eta, p) - lp_norm(eta_bar, p))**2 for eta in landscape)
    return Lambda

    """Topological risk Lambda_i of a single asset over one training window.  [TODO]

    Compose the whole per-asset pipeline:

      1. sub_windows(series, sub_len, shift)          -> sub-series j = 0..T_bar-1
      2. for each: takens_embedding(sub, d, tau)      -> point cloud
      3.           persistence_diagram_h0(cloud)      -> H0 diagram
      4.           finite_pairs(diagram)              -> drop the infinite bar
      5.           persistence_landscape(finite, num_layers, resolution, x_range)
                                                       -> eta^(j)   (shared x_range!)
      6. eta_bar = mean_landscape([eta^(0), ...])     -> mean landscape
      7. Lambda  = sum_j ( lp_norm(eta^(j), p) - lp_norm(eta_bar, p) )^2

    The reference term is lp_norm(eta_bar), the norm of the MEAN landscape --
    not the mean of the norms. Getting that wrong is a silent correctness bug.

    The x_range in step 5 must be the SAME for every sub-window of this asset, or
    step 6 cannot average pointwise. Decide how to fix it (e.g. from the max
    finite death across all this asset's sub-window diagrams) and pass it into
    every persistence_landscape call.

    Note (sum vs average): the paper's prose says "average squared difference"
    but Algorithm 1 writes a sum. Because T_bar is the same for every asset, the
    choice only rescales all Lambda_i by a constant and does not change the
    portfolio. This stub is specified with the sum, per the algorithm.

    Args:
        series: 1-D return series for one asset over the training window.
        sub_len: Sub-window length (paper: 126).
        shift: Sub-window stride (paper: 21).
        d: Embedding dimension (paper: 3).
        tau: Delay (paper: 1).
        p: Landscape-norm order (paper: 1).
        num_layers: Landscape layers (paper: 1).
        resolution: Landscape grid resolution.

    Returns:
        Lambda_i >= 0.
    """
    raise NotImplementedError("implement asset_topological_risk")


def risk_matrix(risks: NDArray[np.float64]) -> NDArray[np.float64]:
    """Assemble the diagonal topological-risk matrix Q = diag(Lambda_i).  [IMPLEMENTED]

    Diagonal because the paper models zero cross-asset topological interaction
    (a stated limitation / future-work item, not an oversight).

    Args:
        risks: 1-D array of per-asset risks Lambda_i, all >= 0.

    Returns:
        ``(n, n)`` diagonal matrix.

    Raises:
        ValueError: If any risk is negative.
    """
    risks = np.asarray(risks, dtype=np.float64)
    if (risks < 0).any():
        raise ValueError("topological risks must be non-negative")
    return np.diag(risks)
