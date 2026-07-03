"""Vietoris-Rips H0 persistence via Gudhi.

Implemented glue. The only file that touches the TDA library, so if you swap
Gudhi for something else later, this is the single place that changes. Gudhi is
imported lazily so the package imports without it installed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Diagram = NDArray[np.float64]  # shape (n, 2): columns are (birth, death)


def persistence_diagram_h0(
    point_cloud: NDArray[np.float64],
    max_edge_length: float = np.inf,
) -> Diagram:
    """0-dimensional Vietoris-Rips persistence diagram of a point cloud.  [IMPLEMENTED]

    Uses Gudhi's ``RipsComplex``. For H0 from Rips, all births are 0 and deaths
    are the filtration values at which connected components merge; exactly one
    component never dies (death = +inf). Use :func:`finite_pairs` to drop it
    before landscaping.

    ``max_dimension=1`` is enough for H0: edges (1-simplices) are what merge
    components, so we do not need to build triangles.

    Args:
        point_cloud: ``(n_points, dim)`` array.
        max_edge_length: Cap on the Rips filtration; ``inf`` builds the full
            filtration.

    Returns:
        A ``(n, 2)`` array of ``(birth, death)`` H0 intervals, one row with
        ``death = inf``.
    """
    import gudhi  # lazy import

    rips = gudhi.RipsComplex(points=point_cloud, max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=1)
    st.compute_persistence()
    return st.persistence_intervals_in_dimension(0)


def finite_pairs(diagram: Diagram) -> Diagram:
    """Drop rows with infinite death.  [IMPLEMENTED]

    The persistence landscape is only defined on finite (birth, death) pairs.

    Args:
        diagram: A ``(n, 2)`` diagram array.

    Returns:
        The sub-array of rows whose death coordinate is finite.
    """
    if diagram.size == 0:
        return diagram.reshape(0, 2)
    return diagram[np.isfinite(diagram[:, 1])]
