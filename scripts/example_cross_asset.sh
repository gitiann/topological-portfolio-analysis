#!/usr/bin/env bash
# Fetch a cross-asset basket and compare topological risk against volatility.
# Usage:  bash scripts/example_cross_asset.sh [START] [END]
set -euo pipefail

START="${1:-2020-01-01}"
END="${2:-$(date +%F)}"
OUT=data/cross_asset

mkdir -p "$OUT"

python -c "import yfinance" 2>/dev/null || {
    echo "yfinance not installed. Run: pip install yfinance" >&2; exit 1; }

echo "Fetching $START to $END ..."
python scripts/fetch_prices.py --outdir "$OUT" --start "$START" --end "$END" --tickers \
  BTC-USD ETH-USD SOL-USD \
  'GC=F' 'SI=F' 'NG=F' 'ZC=F' \
  'EURUSD=X' 'GBPUSD=X' 'USDJPY=X' \
  '^GSPC' '^FTSE' '^N225' \
  TLT HYG KO TSLA \
  VOD.L 7203.T SAP.DE

echo
python scripts/compare_risk_vs_vol.py --files "$OUT"/*.csv