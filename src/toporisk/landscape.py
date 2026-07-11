"""Persistence landscapes and their L^p norms.

This is the analytical core. Two functions are stubs (``persistence_landscape``,
``lp_norm``); one is implemented (``mean_landscape``), but read *why* it can be
implemented as a plain pointwise mean — that constraint dictates the shape of
the landscape API and is the whole reason a per-sub-window scalar is the wrong
return type.

The reference in the risk formula is the norm of the MEAN landscape,
||mean_j eta^(j)||, not the mean of the norms. Since a norm is convex, not
linear, ||mean|| != mean(||.||). So you must be able to average landscapes
*pointwise* before norming, which means every sub-window's landscape must be
sampled on the SAME x-grid. A function returning only a float per sub-window has
already destroyed the information needed to do that.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from toporisk.topology import Diagram

# A sampled landscape: shape (num_layers, resolution). Every landscape in a
# comparison MUST share the same (x_min, x_max, resolution) so they can be
# averaged pointwise.
Landscape = NDArray[np.float64]


def persistence_landscape(
    diagram: Diagram,
    num_layers: int = 1,
    resolution: int = 500,
    x_range: tuple[float, float] | None = None
) -> Landscape:
    """Sampled persistence landscape of a (finite) diagram.  [IMPLEMENTED]

    eta_k (t) = k-th largest value of Lambda_{(b,d)}(t) over all points in the diagram.
    num_layers = number of landscape layers eta_1, ..., eta_{num_layers} to sample (default is 1 as per paper).

    The landscape turns each finite pair (b, d) into a tent function

        Lambda_{(b,d)}(t) = max( 0, min(t - b, d - t) ),

    and the k-th landscape layer lambda_k(t) is the k-th largest tent value at t.
    Sample lambda_1..lambda_{num_layers} on a grid of ``resolution`` points over
    ``x_range`` and return them stacked as shape ``(num_layers, resolution)``.

    Paper values: ``num_layers = 1`` (k = 1) is what Goel-Sharma-Kanniainen use.

    Critical design constraint: ``x_range`` must be *fixed* (passed in, shared
    across all sub-windows of an asset), NOT inferred per-diagram. Otherwise two
    landscapes live on different grids and cannot be averaged pointwise, which
    breaks the mean-landscape step. If ``x_range`` is None, decide on a sensible
    shared default and document it -- but the caller in ``risk.py`` should pin it.

    You may build the tent functions yourself (good for understanding), or use
    ``gudhi.representations.Landscape`` with a fixed ``sample_range``. Either way,
    handle the empty diagram: return zeros of shape ``(num_layers, resolution)``.

    Args:
        diagram: A finite H0 diagram, ``(n, 2)``.
        num_layers: Number of landscape layers to sample.
        resolution: Number of grid points.
        x_range: ``(x_min, x_max)`` sampling range; must be shared across a
            comparison set.

    Returns:
        Array of shape ``(num_layers, resolution)``.
    """

    if diagram is None or diagram.size == 0:
        return np.zeros((num_layers, resolution))
    if x_range is None:
        x_range = (0.0, float(diagram[:, 1].max()))

    tent_values_at_t = np.zeros((diagram.shape[0], resolution))
    for i in range(diagram.shape[0]):
        birth = diagram[i, 0]
        death = diagram[i, 1]
        if birth > death:
            raise ValueError(f"Invalid diagram point: {diagram[i]} (birth > death)")
        if birth < x_range[0] or death > x_range[1]:
            raise ValueError(f"Diagram point {diagram[i]} is outside the x_range {x_range}")
        for t in range(resolution):
            tick = x_range[0] + t * (x_range[1] - x_range[0]) / (resolution - 1)
            tent_values_at_t[i, t] = max(0.0, min(tick - birth, death - tick))

    eta = np.zeros((num_layers, resolution))
    for t in range(resolution):
        sorted_tent_values = np.sort(tent_values_at_t[:, t])[::-1]
        for k in range(num_layers):
            if k < len(sorted_tent_values):
                eta[k, t] = sorted_tent_values[k]
            else:
                eta[k, t] = 0.0

    return eta


def lp_norm(landscape: Landscape, dx: float, p: float=1.0) -> float:
    """L^p norm of a sampled landscape.  [IMPLEMENTED]

    Treating the sampled landscape as (approximations of) functions, the L^p
    norm stacked over layers is

        ( sum_k sum_t |lambda_k(t)|^p * dx )^(1/p).

    ``dx`` is the grid spacing (x_range width / (resolution - 1)); it makes the
    sum a Riemann approximation of the integral rather than a raw sum, so norms
    computed at different resolutions stay comparable.

    Paper value: ``p = 1``.

    Args:
        landscape: Sampled landscape, ``(num_layers, resolution)``.
        p: Norm order.
        dx: Grid spacing.

    Returns:
        The scalar L^p norm (0.0 for an all-zero landscape).
    """
    landscape = np.asarray(landscape, dtype=np.float64)
    if landscape.size == 0:
        return 0.0
    if p <= 0:
        raise ValueError("p must be positive")
    return float(np.sum(np.abs(landscape) ** p) * dx) ** (1.0 / p)


def mean_landscape(landscapes: list[Landscape]) -> Landscape:
    """Pointwise mean of several landscapes.  [IMPLEMENTED]

    This is a plain elementwise average -- but only because every input is
    sampled on the same grid (see the module docstring and the ``x_range``
    constraint in :func:`persistence_landscape`). If the shapes disagree this
    raises, which is the correct failure: it means the landscapes were not
    sampled comparably.

    Args:
        landscapes: Non-empty list of equally-shaped sampled landscapes.

    Returns:
        The pointwise mean landscape, same shape as each input.

    Raises:
        ValueError: If the list is empty or the shapes disagree.
    """
    if not landscapes:
        raise ValueError("need at least one landscape to average")
    shapes = {ls.shape for ls in landscapes}
    if len(shapes) != 1:
        raise ValueError(f"landscapes must share a shape to average pointwise; got {shapes}")
    return np.mean(np.stack(landscapes, axis=0), axis=0)
