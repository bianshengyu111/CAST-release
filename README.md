# CAST: Congestion-Aware Spectral Topology for LEO Satellite Networks

Reference implementation for

> **"Topology Optimization for Low Earth Orbit Satellite Networks: A Congestion-Aware
> Spectral Graph Approach"** — Bian, Huang & Bao, *International Journal of Satellite
> Communications and Networking* (manuscript IJSCN-5182484).

This is the topology/routing decision layer plus the traffic model, KPI aggregation and
analysis scripts that produce the paper's tables and figures. It is self-contained: it
runs with **no external simulator and no network access** (the default propagator is
analytic; SGP4/TLE propagation is supported but optional).

## Layout

```
CAST-release/
├── src/
│   ├── const.py                    all physical / system constants
│   ├── constellation/walker.py     Walker-delta 72x20, 550 km / 53 deg, 1401 active
│   ├── topology/visibility.py      candidate ISLs: d <= 5000 km + Earth-occlusion test
│   ├── topology/access_mapping.py  city -> access star (highest elevation >= 10 deg)
│   ├── topology/cast.py            CAST: alternating optimisation (Algorithm 1)
│   ├── topology/baselines.py       six baselines
│   ├── spectral/fiedler.py         Fiedler vector: exact Lanczos + double-sweep BFS
│   ├── routing/congestion_aware.py Eq. (2) weights, residual capacity per assignment
│   ├── traffic/gravity.py          28-city gravity model, 500-5000 flows
│   ├── simulate/discrete_event.py  packet-level queueing, loss, delay, jitter
│   ├── metrics/kpi.py              11-KPI composite score
│   └── sim/runner.py               run configuration
├── scripts/                        run_single, run_experiments, run_ablation,
│                                   make_tables, make_figures
├── results/
│   ├── raw_kpi_observations.csv    500 raw per-run KPI observations
│   ├── weight_elicitation.csv      per-KPI expert-elicitation scores behind Table 8
│   ├── tables/, tables_full/       generated tables (CSV)
│   └── figures/                    generated figures
└── tests/test_smoke.py
```

## Requirements

Python 3.10+, `pip install -r requirements.txt` (numpy, scipy, matplotlib).

## Quick start

```bash
python tests/test_smoke.py                                            # 10 tests, seconds
python scripts/run_single.py --algorithm CAST --flows 5000 --seed 1
python scripts/run_single.py --algorithm CAST --small --slots 3        # seconds
python scripts/run_experiments.py --quick                              # smoke matrix
python scripts/make_tables.py --raw results/raw_kpi_observations.csv
```

## Reproducing the paper's tables and figures

```bash
# Table 4 / 5 : 7 algorithms x 5 flow counts x 10 seeds x 60 slots
python scripts/run_experiments.py --jobs 8
python scripts/make_tables.py

# Table 6 : congestion regime (single flow count)
python scripts/run_experiments.py --jobs 8 --flows 5000 --seeds 10

# Table 8 : ablation
python scripts/run_ablation.py --jobs 8

# Figures 3-4
python scripts/make_figures.py
```

`--jobs N` parallelises across cores; runs are independent per seed. Outputs land in
`results/`. The full Table 4 matrix takes hours on a workstation.

## Algorithm

CAST scores every inactive candidate edge with

```
S_ij = eta1 * G_tr + eta2 * G_lb + eta3 * G_sp - eta4 * C_dist
```

where `G_tr` is the incident demand at `i` and `j`, `G_lb` the maximum utilisation at
each endpoint, `G_sp = (q_i - q_j)^2` the Fiedler-vector marginal gain of the graph
Laplacian, and `C_dist = d_ij / d_max`. Each iteration re-routes flows with
congestion-aware shortest paths (penalty `Psi(rho) = rho / (1 - rho + eps)`), recomputes
the Fiedler vector, re-scores, and adds the best edges subject to `Delta_max = 6` and
`S_ij > delta = 0.05`. Topology and routing are therefore solved **jointly**, by
alternation rather than sequentially. All constants are named in `src/const.py`.

## Notes on scope

* In the original study the packet-level physics ran on the **StarPerf 2.0**
  discrete-event engine, which is obtained from its authors and is **not** redistributed
  here. This release substitutes an equivalent, documented finite-buffer FIFO queueing
  model so the pipeline is runnable end to end; `src/simulate/discrete_event.py` also
  contains a genuine per-packet simulator for validation. The analytic model reproduces
  the qualitative mechanism (a few percent loss at ~82 % slot-average peak utilisation);
  its absolute loss/tail values are **not** guaranteed to match the paper's decimals.
* `results/weight_elicitation.csv` holds the raw Delphi-style panel scores (three
  panellists, three rounds, eleven KPIs) from which the weights in Table 8 were set. The
  panellists are anonymised as E1--E3 with their years of experience and role recorded in
  the file header. No inter-rater agreement statistic is claimed in the manuscript: with
  three raters over eleven items Kendall's W is sensitive to the tie-handling convention,
  so the paper argues robustness to the weighting choice instead (Tables 9 and 11).
* Composite scores are computed from full-precision per-run KPI values, before the
  rounding shown in the paper's tables.
* The two learning-based baselines follow the **published reward structure** of
  Wang-MADRL (hop count + energy) and DGL-JCR (connectivity + routing), not the exact
  network architectures, and are evaluated out-of-distribution as the paper states.
* **This reference evaluator does not reproduce the paper's exact ranking.** In a sample
  run (5,000 flows, 1 seed, 3 slots, composite score): DGL-JCR 0.994, Wang-MADRL 0.973,
  CAST 0.880, 6-nearest 0.767, Nie-DTC-DPSO 0.761, Triangle 0.595, Grid+ 0.193. CAST
  leads every **non-learning** baseline on loss (0.85 % vs 9.05 % for 6-nearest), delay,
  energy (41.8 vs 62.7 kWh) and peak utilisation (88 % vs 97.2 %) — the paper's central
  claim — but the two learning *surrogates* rank above it, whereas the manuscript ranks
  CAST first. Both reasons are structural: the surrogates place all six edges per node
  directly from the demand signal, which is stronger than the original networks this
  release does not reproduce; and the manuscript's own caveat applies, namely that the
  RL comparison is confounded by an information asymmetry and that absolute KPI values
  depend on the packet-level engine.

## Data availability

Constellation geometry uses the analytic Walker-delta propagator. Real trajectories can
be derived from public TLE data (CelesTrak, <https://celestrak.org/NORAD/elements/>) by
passing a TLE file to `WalkerDelta(propagator="sgp4", tle_path=...)`. StarPerf 2.0 is
obtained from its authors and is not bundled. The release accompanying the manuscript is
publicly available at <https://github.com/bianshengyu111/CAST-release>, and this URL is
cited in the paper's Data Availability Statement.

## Citation

```bibtex
@article{bian2026cast,
  title   = {Topology Optimization for Low Earth Orbit Satellite Networks:
             A Congestion-Aware Spectral Graph Approach},
  author  = {Bian, Shengyu and Huang, Junjie and Bao, Yifei},
  journal = {International Journal of Satellite Communications and Networking},
  year    = {2026},
  note    = {Manuscript IJSCN-5182484}
}
```

## License

MIT — see `LICENSE`.
