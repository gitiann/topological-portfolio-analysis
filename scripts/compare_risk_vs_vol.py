# scripts/compare_risk_vs_vol.py
import argparse
import numpy as np
import pandas as pd
from toporisk.data import simple_returns
from toporisk.risk import asset_topological_risk

def load_close(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)  # silences the warning too
    col = "Close" if "Close" in df.columns else df.columns[-1]
    return df[[col]].rename(columns={col: "price"})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    args = ap.parse_args()

    rows = []
    for path in args.files:
        prices = load_close(path)
        rets = simple_returns(prices)["price"].to_numpy()
        lam = asset_topological_risk(rets)
        vol = rets.std(ddof=1) * np.sqrt(252)      # annualized volatility
        rows.append((path.split("/")[-1], lam, vol))

    df = pd.DataFrame(rows, columns=["asset", "Lambda", "AnnVol"])
    # rank each independently; if topology == volatility, the ranks match exactly
    df["Lambda_rank"] = df["Lambda"].rank()
    df["Vol_rank"] = df["AnnVol"].rank()
    # Spearman correlation of the two rankings across assets
    print(df.to_string(index=False))
    if len(df) > 2:
        rho = df["Lambda"].corr(df["AnnVol"], method="spearman")
        print(f"\nSpearman(Lambda, AnnVol) = {rho:.3f}")

if __name__ == "__main__":
    main()