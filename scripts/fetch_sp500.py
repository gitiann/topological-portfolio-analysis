"""Download S&P500 historical adjusted close series for ticker list.

This script uses `yfinance` to fetch daily adjusted close prices for a
list of tickers (default: SPY as S&P500 proxy). It writes CSVs into
`data/` for use by the model.

Usage:
    python scripts/fetch_sp500.py --tickers SPY AAPL MSFT --start 2018-01-01 --end 2023-12-31
"""
import argparse
import pathlib
import sys

try:
    import yfinance as yf
except Exception:
    print("yfinance is required. Install with: pip install yfinance pandas", file=sys.stderr)
    raise

import pandas as pd


def fetch_and_save(tickers, start, end, outdir):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        print(f"Downloading {t}...")
        data = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if data.empty:
            print(f"No data for {t}, skipping.")
            continue

        close = data["Close"]
        # yf may return a 1-column DataFrame (MultiIndex) or a Series; force Series
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        close.name = "price"

        csv_path = outdir / f"{t}.csv"
        close.to_csv(csv_path)              # writes clean: date(index),price
        print(f"Saved {csv_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tickers', nargs='+', default=['SPY'], help='Tickers to download')
    p.add_argument('--start', default='2018-01-01')
    p.add_argument('--end', default='2023-12-31')
    p.add_argument('--outdir', default='data')
    args = p.parse_args()
    fetch_and_save(args.tickers, args.start, args.end, args.outdir)


if __name__ == '__main__':
    main()
