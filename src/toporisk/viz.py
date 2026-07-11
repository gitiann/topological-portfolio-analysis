"""Plotting helpers. Implemented glue."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from toporisk.landscape import Landscape


def plot_weights(
    assets: list[str],
    weights: NDArray[np.float64],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar chart of portfolio weights.

    Args:
        assets: Asset labels.
        weights: Weight per asset (same order as ``assets``).
        ax: Optional axes.

    Returns:
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.bar(assets, weights)
    ax.set_ylabel("weight")
    ax.set_title("Minimum topological-risk portfolio")
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    return ax


def plot_landscape(landscape: Landscape, ax: plt.Axes | None = None) -> plt.Axes:
    """Line plot of a sampled persistence landscape (one line per layer).

    Args:
        landscape: Sampled landscape, ``(num_layers, resolution)``.
        ax: Optional axes.

    Returns:
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    grid = np.arange(landscape.shape[1])
    for k in range(landscape.shape[0]):
        ax.plot(grid, landscape[k], label=f"lambda_{k + 1}")
    ax.set_xlabel("grid index")
    ax.set_ylabel("landscape value")
    ax.legend(loc="best")
    return ax
