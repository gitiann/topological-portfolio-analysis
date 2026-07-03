"""Price loading and return computation.

Fully implemented plumbing. Note the paper uses *simple* returns, not log
returns, so that is the default here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_prices(path: str | Path) -> pd.DataFrame:
    """Load a wide price panel from CSV.

    Expects a ``date`` column (parsed as the index) and one price column per
    asset.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame indexed by ``DatetimeIndex``, one float column per asset,
        sorted ascending, fully-empty rows dropped.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If there is no ``date`` column.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.dropna(how="all")


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple (arithmetic) returns: r_t = (P_t - P_{t-1}) / P_{t-1}.

    This is the return definition used in Goel-Sharma-Kanniainen. The first row
    is dropped.

    Args:
        prices: Wide price panel (rows = dates, columns = assets), strictly positive.

    Returns:
        Simple-returns DataFrame, one fewer row than ``prices``.

    Raises:
        ValueError: If any price is non-positive.
    """
    if (prices <= 0).to_numpy().any():
        raise ValueError("prices must be strictly positive")
    return prices.pct_change().dropna(how="all")


def asset_series(returns: pd.DataFrame, asset: str) -> np.ndarray:
    """Extract one asset's return series as a 1-D float array, NaNs dropped.

    Args:
        returns: Returns DataFrame.
        asset: Column name.

    Returns:
        1-D float64 array of that asset's returns.
    """
    return returns[asset].dropna().to_numpy(dtype=np.float64)
