"""Command-line interface.

Example
-------
    toporisk build --prices data/sample_prices.csv --window 252 \
        --sub-len 126 --shift 21 --out figures/weights.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from toporisk.data import asset_series, load_prices, simple_returns
from toporisk.portfolio import min_topological_risk_portfolio
from toporisk.risk import asset_topological_risk, risk_matrix
from toporisk.viz import plot_weights


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toporisk", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build the minimum topological-risk portfolio")
    build.add_argument("--prices", required=True, type=Path, help="CSV panel with a 'date' column")
    build.add_argument("--window", type=int, default=252, help="training window length (days)")
    build.add_argument("--sub-len", type=int, default=126, help="sub-window length (paper: 126)")
    build.add_argument("--shift", type=int, default=21, help="sub-window shift (paper: 21)")
    build.add_argument("--d", type=int, default=3, help="embedding dimension (paper: 3)")
    build.add_argument("--tau", type=int, default=1, help="embedding delay (paper: 1)")
    build.add_argument("--out", type=Path, default=None, help="save the weights bar chart here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "build":
        prices = load_prices(args.prices)
        returns = simple_returns(prices)
        # use the last `window` days as the training window
        returns = returns.tail(args.window)
        assets = list(returns.columns)

        try:
            risks = np.array(
                [
                    asset_topological_risk(
                        asset_series(returns, a),
                        sub_len=args.sub_len,
                        shift=args.shift,
                        d=args.d,
                        tau=args.tau,
                    )
                    for a in assets
                ]
            )
            Q = risk_matrix(risks)
            weights = min_topological_risk_portfolio(Q)
        except NotImplementedError as exc:
            print(f"pipeline is not fully implemented yet: {exc}", file=sys.stderr)
            return 2

        for a, w in zip(assets, weights, strict=True):
            print(f"{a:>12}  {w:8.4f}")

        plot_weights(assets, weights)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.out, dpi=150, bbox_inches="tight")
            print(f"saved {args.out}")
        else:
            plt.show()
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
