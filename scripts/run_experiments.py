#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the full experiment matrix and dump raw per-run KPIs to CSV.

Default matrix (manuscript): 7 algorithms x 5 flow counts x 10 seeds = 350 runs,
each 60 slots.  Results land in ``results/raw_kpi_observations.csv`` and feed
``make_tables.py`` / ``make_figures.py``.

Use ``--jobs N`` to parallelise across processes (runs are independent per seed).
Use ``--quick`` for a reduced matrix that finishes in minutes and validates the
pipeline end to end.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constellation.walker import walker_72x20                  # noqa: E402
from src.metrics.kpi import ConfigKPIs, KPI_DEFS                   # noqa: E402
from src.sim.runner import RunConfig, run_configuration            # noqa: E402
from src.topology.baselines import ALGORITHMS                      # noqa: E402

FIELDS = ["algorithm", "flows", "seed", "throughput_gbps", "reachability_frac",
          "loss_rate_pct", "mean_delay_ms", "tail_delay_ms", "jitter_ms",
          "business_stability_frac", "peak_util_frac", "load_imbalance_frac",
          "energy_kwh", "lambda2"]


def _worker(job):
    algorithm, n_flows, seed, slots, small, ca2 = job
    if small:
        walker = walker_72x20(planes=10, sats_per_plane=20, active=200)
        planes, spp = 10, 20
    else:
        walker = walker_72x20()
        planes, spp = 72, 20
    cfg = RunConfig(algorithm=algorithm, n_flows=n_flows, seed=seed, n_slots=slots,
                    ca2=ca2)
    k: ConfigKPIs = run_configuration(cfg, walker, planes, spp)
    return [algorithm, n_flows, seed, k.throughput_gbps, k.reachability_frac,
            k.loss_rate_pct, k.mean_delay_ms, k.tail_delay_ms, k.jitter_ms,
            k.business_stability_frac, k.peak_util_frac, k.load_imbalance_frac,
            k.energy_kwh, k.lambda2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--slots", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--flows", type=int, nargs="*", default=[500, 1000, 2000, 4000, 5000])
    ap.add_argument("--algorithms", nargs="*", default=ALGORITHMS)
    ap.add_argument("--quick", action="store_true",
                    help="reduced matrix: 2 algorithms x 2 flow counts x 2 seeds, "
                         "200 satellites")
    ap.add_argument("--out", default="results/raw_kpi_observations.csv")
    args = ap.parse_args()

    if args.quick:
        algorithms = ["CAST", "6-nearest"]
        flows = [500, 2000]
        seeds = 2
        slots = min(args.slots, 5)
        small = True
    else:
        algorithms, flows, seeds, slots, small = args.algorithms, args.flows, args.seeds, args.slots, False

    jobs = list(itertools.product(algorithms, flows, range(1, seeds + 1),
                                  [slots], [small], [4.0]))
    print(f"{len(jobs)} runs; jobs={args.jobs}")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_worker, j): j for j in jobs}
            for n, fut in enumerate(as_completed(futs), 1):
                rows.append(fut.result())
                print(f"  {n}/{len(jobs)} done", flush=True)
    else:
        for n, j in enumerate(jobs, 1):
            rows.append(_worker(j))
            print(f"  {n}/{len(jobs)}  {j[0]} flows={j[1]} seed={j[2]}", flush=True)

    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        w.writerows(rows)
    print(f"wrote {out.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
