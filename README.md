# CAST: Congestion-Aware Spectral Topology for LEO Satellite Networks

Reference implementation for

> **"Topology Optimization for Low Earth Orbit Satellite Networks: A Congestion-Aware
> Spectral Graph Approach"** — Bian, Huang & Bao, *International Journal of Satellite
> Communications and Networking* (manuscript IJSCN-5182484).

This release covers the proposed method (CAST), the four non-learning baselines, the
routing / traffic / KPI stack, and the analysis scripts that generate the paper's tables
and figures. It is self-contained: it runs with **no external simulator and no network
access** (the default propagator is analytic; SGP4/TLE propagation is supported but
optional).

The manuscript's two **learning-based baselines** — Wang-MADRL [19] and DGL-JCR [27] —
are *not* reimplemented here. They are external methods with their own networks,
training pipelines and hyper-parameters; a stand-in reimplementation would not be the
method that was compared, so the release stops at CAST plus every baseline that needs no
learning. Consequently the seven-algorithm ranking of Table 4 is **not** reproduced end
to end by this code, and absolute KPI values are not guaranteed to match the
manuscript's decimals (see *Notes on scope* and *Revision history*).

## Layout

```
CAST-release/
├── src/
│   ├── const.py                    all physical / system constants
│   ├── constellation/walker.py     Walker-delta 72x20, 550 km / 53 deg, 1401 active
│   ├── topology/visibility.py      candidate ISLs: d <= 5000 km + Earth-occlusion test
│   ├── topology/access_mapping.py  city -> access star (highest elevation >= 10 deg)
│   ├── topology/cast.py            CAST: alternating optimisation (Algorithm 1)
│   ├── topology/baselines.py       four non-learning baselines
│   ├── spectral/fiedler.py         Fiedler vector: exact Lanczos + double-sweep BFS
│   ├── routing/congestion_aware.py Eq. (2) weights, residual capacity per assignment
│   ├── traffic/gravity.py          28-city gravity model, 500-5000 flows
│   ├── simulate/discrete_event.py  packet-level queueing, loss, delay, jitter
│   ├── metrics/kpi.py              11-KPI composite score
│   └── sim/runner.py               run configuration
├── scripts/                        run_single, run_experiments, run_ablation,
│                                   make_tables, make_figures
├── results/
│   ├── weight_elicitation.csv      per-KPI expert-elicitation scores behind Table 8
│   ├── raw_kpi_observations.csv    generated: raw per-run KPI observations
│   ├── tables/, tables_full/       generated tables (CSV)
│   └── figures/                    generated figures
└── tests/test_smoke.py
```

`results/weight_elicitation.csv` is the only tracked file under `results/`; the KPI dumps
and the generated tables/figures are produced by the scripts above and are git-ignored.

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

## Running the exposed configuration

```bash
# five algorithms x 5 flow counts x 10 seeds x 60 slots (the manuscript's protocol)
python scripts/run_experiments.py --jobs 8
python scripts/make_tables.py

# congestion regime (single flow count)
python scripts/run_experiments.py --jobs 8 --flows 5000 --seeds 10

# ablation of the CAST score terms
python scripts/run_ablation.py --jobs 8

# figures from the raw KPI dump
python scripts/make_figures.py
```

`--jobs N` parallelises across cores; runs are independent per seed. Outputs land in
`results/`. The full matrix takes hours on a workstation. Tables 4-5 of the manuscript
cover seven algorithms; the five covered here are run with the same protocol, so the
resulting tables are the release's own run of that subset, not the manuscript's tables.

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
* **Scope of the comparison.** The evaluator implements CAST and the four non-learning
  baselines (6-nearest, Grid+, Triangle, Nie-DTC-DPSO). The two learning-based baselines
  of the manuscript are external methods and are not reimplemented, so the seven-algorithm
  ranking of Table 4 is not reproduced end to end here. Within the covered set the composite
  score ranks CAST first, then 6-nearest, Nie-DTC-DPSO, Triangle and Grid+: CAST takes the
  lowest mean and tail delay, the lowest energy and the lowest load imbalance, while
  Triangle reaches a lower loss and a marginally lower peak utilisation at six times the
  delay and seven times the energy. These are the release's own numbers in the exposed
  configuration, not the manuscript's.

## Revision history

### 2026-09-27 — scope and algorithm corrections

* **The two learning-based baselines were removed from the release.** They were
  stand-ins that allocated all six terminals per node directly from the offered demand
  matrix of the slot — an information set no deployed policy has — and their ranking
  against CAST therefore said more about the stand-in than about the method. Rather than
  ship a surrogate whose behaviour cannot be defended, the release now covers only CAST
  and the four non-learning baselines. The manuscript's caveat that the reinforcement
  learning comparison is confounded still stands.
* **CAST's 1-swap path had two defects, both fixed.** With the degree budget saturated,
  a candidate whose *both* endpoints were full evicted one link at each end and added
  one, so every accepted candidate shrank the active link set by one; and a swap that
  succeeded at the first endpoint but found nothing evictable at the second returned
  without adding the edge, dropping a link for nothing. The first is now rejected
  outright and the second is checked before either endpoint is touched, so a swap is
  atomic: no link is dropped by a swap that does not complete. The effect is visible over
  long horizons — without the fixes the active link count fell from 4,123 to 1,076 over
  ten slots, with peak utilisation above 100 % from slot six onward. The count still
  declines in this configuration (4,199 to 1,116 over ten slots at 5,000 flows); the
  passive-removal rule below is the main reason, and it is a property of the rule rather
  than of the swap.
* **The passive removal is applied at face value.** Algorithm 1 removes dynamic ISLs that
  carry no flow, and this implementation applies that to *every* idle link at the end of
  each slot. In the 5,000-flow configuration that sheds far more links per slot than the
  per-slot idle churn of the manuscript's ISL-switching table (~6.5 links), because
  congestion-aware shortest paths leave a large part of the active set unused in any one
  slot. Active link counts and absolute KPIs over long runs are therefore not expected to
  match the manuscript's: the release is a readable, runnable reference for the topology
  and routing rules, not a numerical reproduction of Tables 4-7.

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
