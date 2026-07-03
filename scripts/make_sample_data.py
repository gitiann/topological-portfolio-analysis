"""Generate a small synthetic price panel so the pipeline is runnable.

Not real market data. Assets are given deliberately different volatilities so
that, once you implement the pipeline, their topological risks Lambda_i differ
and the optimiser produces non-uniform weights. Replace with real S&P 500
prices for actual reproduction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(7)
    n_days = 400
    dates = pd.date_range("2019-01-01", periods=n_days, freq="B")

    # Six assets with a spread of volatilities (annualised-ish, per-day scale).
    vols = {
        "LOWVOL_A": 0.004,
        "LOWVOL_B": 0.006,
        "MIDVOL_C": 0.010,
        "MIDVOL_D": 0.014,
        "HIVOL_E": 0.022,
        "HIVOL_F": 0.030,
    }
    cols = {}
    for name, vol in vols.items():
        shocks = rng.normal(scale=vol, size=n_days)
        cols[name] = 100 * np.exp(np.cumsum(shocks))

    df = pd.DataFrame(cols, index=dates)
    df.index.name = "date"

    out = Path(__file__).resolve().parents[1] / "data" / "sample_prices.csv"
    df.to_csv(out)
    print(f"wrote {out} ({len(df)} rows, {len(vols)} assets)")


if __name__ == "__main__":
    main()
