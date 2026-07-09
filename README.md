# toporisk — topological risk portfolios

Reproduction skeleton for **Goel, Sharma & Kanniainen, "Class of topological
portfolios: Are they better than classical portfolios?" (2026)**, arXiv:2601.03974
(pre-accepted, *Financial Innovation*).

> **Status: skeleton.** The glue (data, windowing, Gudhi H0 persistence, the risk
> matrix, the CLI) is implemented and clean under `pytest`/`mypy`/`ruff`. The five
> pieces of mathematics are specified stubs with tests that `xfail` until you
> implement them. This is deliberate — the point is the code you write.

## The model

For each asset *independently*: take its simple-return series, cut it into
overlapping sub-windows, Takens-embed each sub-window into a point cloud, compute
the H0 (connected-components) persistence landscape of each cloud, and measure how
much the landscape's L¹ norm *varies* across sub-windows. That variability is the
asset's **topological risk** Λᵢ. Assets are then coupled only through a diagonal
risk matrix `Q = diag(Λᵢ)` and a long-only quadratic program that minimises
`wᵀQw`. There is no cross-asset topology in this model by design (the paper
flags cross-asset coupling as future work).

Fixed parameters, per the paper (verify against your copy): simple returns;
`d = 3`, `τ = 1`; sub-window length `T̃ = 126`, shift `h = 21`; H0 only; landscape
layer `k = 1`; norm order `p = 1`.

## Layout

```
src/toporisk/
  data.py        load prices, simple returns                 [implemented]
  embedding.py   sub_windows [done] | takens_embedding [TODO]
  topology.py    Gudhi H0 diagram, finite_pairs              [implemented]
  landscape.py   mean_landscape [done] | persistence_landscape, lp_norm [TODO]
  risk.py        risk_matrix [done] | asset_topological_risk [TODO]
  portfolio.py   min_topological_risk_portfolio              [TODO]
  cli.py         `toporisk build ...`                        [implemented]
  viz.py         weight + landscape plots                    [implemented]
tests/           pytest; stubs marked xfail(strict=True)
scripts/         make_sample_data.py
```

## Install & run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,qp]"
python scripts/make_sample_data.py
pytest                       # 17 pass, 10 xfail on a fresh checkout
toporisk build --prices data/sample_prices.csv --out figures/weights.png
```

The CLI runs once you've implemented the stubs; before that it exits with a clear
"not implemented" message.

## Roadmap

Each has an `xfail(strict=True)` test. Correct implementation flips it to a
pass (reported as an unexpected pass); then delete the marker.

1. [IMPLEMENTED]`embedding.takens_embedding` — you already derived it: length-`T̃` series,
   `d=3, τ=1` → `T̃-2` points in R³. The toy test pins the exact vectors from the
   `[1,-2,3,-1,2]` example; the shape test pins `124×3`.
2. [IMPLEMENTED]`landscape.persistence_landscape` — tent functions → sampled landscape on a
   **fixed shared grid**. Read the module docstring on *why* the grid must be
   shared before you write it.
3. [IMPLEMENTED]`landscape.lp_norm` — discrete Lᵖ norm of a sampled landscape (`p=1`).
4. [IN_PROGRESS]`risk.asset_topological_risk` — the capstone. Composes 1–3 over the
   sub-windows. The reference is `‖mean landscape‖`, **not** `mean(‖landscape‖)`.
   This is the whole reason the landscape API returns an object, not a float.
5. [TODO]`portfolio.min_topological_risk_portfolio` — the convex QP.
