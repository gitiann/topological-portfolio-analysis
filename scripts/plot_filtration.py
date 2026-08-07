"""Vietoris-Rips filtration + H0 barcode from a real price CSV.

Usage:
    python scripts/plot_filtration.py --file data/AAPL.csv --n-points 18
    python scripts/plot_filtration.py --file data/AAPL.csv --start 250 --n-points 20

Reads a CSV with a date index and a 'price' column (the format fetch_sp500.py
writes), converts to simple returns, Takens-embeds a short slice into R^2, and
draws the filtration with the H0 barcode it produces.

Uses d=2 purely so the cloud is drawable; the model itself uses d=3.
Keep --n-points around 18-20: more makes the edges a hairball and the colour
palette starts recycling, so separate components look identical.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform


def load_returns(path: str) -> np.ndarray:
    """Simple returns from a date-indexed CSV with a price column."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    col = "price" if "price" in df.columns else df.columns[-1]
    prices = pd.to_numeric(df[col], errors="coerce").dropna()
    if (prices <= 0).any():
        raise ValueError(f"{path} contains non-positive prices")
    return prices.pct_change().dropna().to_numpy()


def takens_2d(series: np.ndarray, tau: int = 1) -> np.ndarray:
    """Delay embedding into R^2: row j = (s[j], s[j+tau])."""
    n = len(series) - tau
    if n < 1:
        raise ValueError("series too short to embed")
    return np.column_stack([series[:n], series[tau:n + tau]])


def components_at(D: np.ndarray, r: float):
    A = csr_matrix((D <= r) & (D > 0))
    return connected_components(A, directed=False)


def h0_deaths(cloud: np.ndarray) -> np.ndarray:
    """Finite H0 death radii (the merge radii), via Gudhi."""
    import gudhi
    st = gudhi.RipsComplex(points=cloud).create_simplex_tree(max_dimension=1)
    st.compute_persistence()
    dgm = st.persistence_intervals_in_dimension(0)
    return np.sort(dgm[np.isfinite(dgm[:, 1])][:, 1])


def make_figure(cloud, radii, out, label=""):
    D = squareform(pdist(cloud))
    deaths = h0_deaths(cloud)
    cmap = plt.get_cmap("tab20")

    fig = plt.figure(figsize=(15, 6.6))
    gs = fig.add_gridspec(2, len(radii), height_ratios=[2.6, 1.0], hspace=0.18)

    for k, r in enumerate(radii):
        ax = fig.add_subplot(gs[0, k])
        n_comp, labels = components_at(D, r)
        for (x, y), lab in zip(cloud, labels):
            ax.add_patch(Circle((x, y), r / 2, color=cmap(lab % 20), alpha=0.15, lw=0))
        for i in range(len(cloud)):
            for j in range(i + 1, len(cloud)):
                if D[i, j] <= r:
                    ax.plot(*zip(cloud[i], cloud[j]), color="0.35", lw=0.9, zorder=2)
        ax.scatter(cloud[:, 0], cloud[:, 1], c=[cmap(l % 20) for l in labels],
                   s=40, zorder=3, edgecolors="white", linewidths=0.7)
        ax.set_title(f"radius = {r:.4f}   →   {n_comp} component{'s' if n_comp != 1 else ''}",
                     fontsize=10)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    axb = fig.add_subplot(gs[1, :])
    for i, d in enumerate(deaths):
        axb.plot([0, d], [i, i], lw=2.0, color="0.25", solid_capstyle="butt")
    axb.plot([0, deaths.max() * 1.12], [len(deaths), len(deaths)],
             lw=2.0, color="crimson", solid_capstyle="butt")
    axb.text(deaths.max() * 1.13, len(deaths), " never dies", va="center",
             fontsize=9, color="crimson")
    for r in radii:
        axb.axvline(r, color="steelblue", ls="--", lw=1.0, alpha=0.8)
    axb.set_xlabel("radius (filtration value)")
    axb.set_ylabel("H₀ bars")
    axb.set_title("H₀ barcode: each bar is a component, dying at the radius where it merges. "
                  "Dashed lines mark the panels above.", fontsize=10)
    axb.set_yticks([]); axb.set_xlim(left=0)
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)

    if label:
        fig.suptitle(label, y=1.00, fontsize=12)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}   ({len(deaths)} finite bars + 1 infinite)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", required=True, help="price CSV (date index, 'price' column)")
    p.add_argument("--n-points", type=int, default=18, help="points in the cloud (18-20 reads best)")
    p.add_argument("--start", type=int, default=0, help="offset into the return series")
    p.add_argument("--tau", type=int, default=1)
    p.add_argument("--out", default=None, help="output png (default: figures/<ticker>_filtration.png)")
    args = p.parse_args()

    ticker = Path(args.file).stem
    out = args.out or f"figures/{ticker}_filtration.png"

    rets = load_returns(args.file)
    need = args.n_points + args.tau
    if args.start + need > len(rets):
        raise SystemExit(f"need {need} returns from offset {args.start}, only {len(rets)} available")
    cloud = takens_2d(rets[args.start:args.start + need], tau=args.tau)

    D = squareform(pdist(cloud))
    finite = D[D > 0]
    # four radii spanning the filtration, chosen from the distance distribution
    radii = [np.quantile(finite, q) for q in (0.02, 0.10, 0.25)] + [finite.max() * 0.55]

    make_figure(cloud, radii, out,
                label=f"{ticker}: Takens embedding (d=2, τ={args.tau}) of daily returns, "
                      f"days {args.start}–{args.start + args.n_points}")


if __name__ == "__main__":
    main()