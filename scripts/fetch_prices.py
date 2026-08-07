"""Download daily adjusted close series for a list of tickers.

Uses `yfinance` to fetch split/dividend-adjusted daily closes and writes one
CSV per ticker into the output directory, in the format the model expects:

    date,price
    2020-01-02,235.95

Works for any Yahoo Finance symbol, not just equities:
    equities   AAPL, VOD.L, 7203.T, SAP.DE
    indices    ^GSPC, ^FTSE, ^N225
    crypto     BTC-USD, ETH-USD
    FX         EURUSD=X, GBPUSD=X
    futures    GC=F, CL=F, NG=F

Usage:
    python scripts/fetch_prices.py --tickers SPY AAPL MSFT --start 2018-01-01 --end 2023-12-31
    python scripts/fetch_prices.py --tickers BTC-USD 'GC=F' '^GSPC' --outdir data/cross_asset

Note: quote symbols containing `=` or `^` so the shell does not expand them.
Tickers with no data in the requested window are skipped with a message.
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
