# Status and attribution

This project reproduces the topological portfolio model of Goel, Sharma and
Kanniainen (2026, arXiv:2601.03974). I implemented the mathematical core myself:
the Takens embedding, the persistence landscape and its $L^1$ norm, the asset
topological risk, and the portfolio quadratic program.  The  scaffolding
(packaging, CLI, test harness and plotting) was written with AI assistance. All
31 tests pass, and the pipeline has been validated on real S&P constituent data.

# toporisk — topological risk portfolios

Using topology of data we can avoid model-based estimation errors linked to distributional assumptions or statistical inputs like mean and covariance. A 2026 January paper by Goel, Sharma, Kanniainen (arXiv: https://doi.org/10.48550/arXiv.2601.03974) reports that portfolios constructed using Topological Data Analysis methods outperformed the seven popular portfolio optimization models and two benchmark portfolio strategies, the naive **1/N** portfolio and the S&P 500 market index, in terms of excess mean return and several financial ratios.

This model has significant potential in cases characterized by a high degree of model uncertainty accompanied by substantial estimation errors.

The intuitive idea is that we map a one-dimensional time-series of returns into $\mathbb{R}^3$ space as data-points via Takens' delay embedding. Next we form a simplicial complex using the Vietoris-Rips method by connecting any two data-points that fall within a radius. The radius starts at 0, in which case all points are disconnected (unless they map to the same place, as with constant returns). We continuously increase the radius, and whenever two points fall within it we connect them via an edge. As the radius increases new connections are made, and we are interested in how well the overall shape is connected through the varying radius.

It may happen that at some radius $r_i$ we end up with $n$ distinct point-clouds not connected to each other. This is precisely what the 0-th homology group $H_0$ measures: the sequence of merges, each recorded as a (birth, death) pair, where death is the radius at which two components merge. A tightly clustered cloud merges quickly at small radii; a scattered cloud may have components surviving to large radii. We summarise the topology of our space via a **persistence landscape** (a series of time-dependent functions $\eta_k(t)$, $k = 1, 2, \ldots$).

From a topological point of view, more scattering of the point cloud means less stable returns; a more concentrated cloud, more stable. The amount of scatteredness among return observations over time can be quantified using the $L_1$ norm of the persistence landscape. Therefore this norm can effectively track changes in the state of stock return dynamics without any prior distributional assumptions. Lastly, the persistence landscape forms a Banach space, so the theory of random variables can be applied to it. By definition it involves no parameter and is thus free from parameter tuning and over-fitting risk.[1]

>[1] Bubenik, P., et al. *Statistical topological data analysis using persistence landscapes.* J. Mach. Learn. Res. **16**(1), 77–102 (2015).

## The model

For each asset *independently*: take its simple-return series, cut it into overlapping sub-windows, Takens-embed each sub-window into a point cloud, compute the H0 (0th homology group, connected components) persistence landscape of each cloud, and measure how much the landscape's L¹ norm *varies* across sub-windows. That variability is the asset's **topological risk** Λᵢ. Assets are then coupled only through a diagonal risk matrix `Q = diag(Λᵢ)` and a long-only quadratic program that minimises `wᵀQw`. There is no cross-asset topology in this model, by design (the paper flags cross-asset coupling as future work).

Fixed parameters, per the paper (verify against your copy): simple returns; `d = 3`, `τ = 1`; sub-window length `T̃ = 126`, shift `h = 21`; H0 only; landscape layer `k = 1`; norm order `p = 1`.

## Layout

```src/toporisk/
data.py        load prices, simple returns                 [implemented]
embedding.py   sub_windows, takens_embedding               [implemented]
topology.py    Gudhi H0 diagram, finite_pairs              [implemented]
landscape.py   mean_landscape, persistence_landscape, lp_norm   [implemented]
risk.py        risk_matrix, asset_topological_risk         [implemented]
portfolio.py   min_topological_risk_portfolio              [implemented]
cli.py         toporisk build ...                        [implemented]
viz.py         weight + landscape plots                    [implemented]
tests/           pytest (31 passing)
scripts/         make_sample_data.py
```





## Install & run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,qp]"
python scripts/make_sample_data.py
pytest
toporisk build --prices data/sample_prices.csv --out figures/weights.png
```

## Implementation notes

The pipeline is built and tested end to end. Key functions:

1. `embedding.takens_embedding` — length-`T̃` series, `d=3, τ=1` → `T̃-2` points in $\mathbb{R}^3$.
2. `landscape.persistence_landscape` — tent functions sampled on a fixed shared grid, so landscapes from different sub-windows can be averaged pointwise.
3. `landscape.lp_norm` — discrete Lᵖ norm of a sampled landscape (`p=1`), grid-spacing aware.
4. `risk.asset_topological_risk` — composes 1–3 over the sub-windows. The reference term is `‖mean landscape‖`, **not** `mean(‖landscape‖)`.
5. `portfolio.min_topological_risk_portfolio` — the convex QP; for diagonal `Q` it reduces to inverse-risk weighting, with zero-risk assets excluded.