# -*- coding: utf-8 -*-
"""Checkpointed full-matrix driver with extended per-run columns.

Runs the same protocol as ``run_experiments.py`` but (a) appends each completed row
immediately and resumes from a partial CSV, and (b) records the extra per-run
quantities needed to rebuild the manuscript's congestion and delay-decomposition
tables (mean utilisation, top-5% share, overload fraction, hop count, T_prop /
T_trans / T_queue), aggregated from the same per-slot observations.

Local driver for the revision reruns; the public release entry point remains
``run_experiments.py``.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constellation.walker import walker_72x20                  # noqa: E402
from src.metrics.kpi import ConfigKPIs                             # noqa: E402
from src.sim.runner import RunConfig, run_configuration            # noqa: E402
from src.topology.baselines import ALGORITHMS                      # noqa: E402

FIELDS = ["algorithm", "flows", "seed", "throughput_gbps", "reachability_frac",
          "loss_rate_pct", "mean_delay_ms", "tail_delay_ms", "jitter_ms",
          "business_stability_frac", "peak_util_frac", "load_imbalance_frac",
          "energy_kwh", "lambda2"]

EXTRA_FIELDS = ["mean_util_frac", "top5pct_share", "overload_link_frac",
                "oversub_link_frac", "mean_hops", "n_active_edges",
                "t_prop_ms", "t_trans_ms", "t_queue_ms"]

FIELDS_ALL = FIELDS + EXTRA_FIELDS


def _weighted_mean(vals, weights):
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        return 0.0
    return float(np.average(np.asarray(vals, dtype=float), weights=w))


def _worker(job):
    algorithm, n_flows, seed, slots = job
    walker = walker_72x20()
    cfg = RunConfig(algorithm=algorithm, n_flows=n_flows, seed=seed, n_slots=slots,
                    ca2=4.0)
    obs_list: list = []
    k: ConfigKPIs = run_configuration(cfg, walker, 72, 20, observations_out=obs_list)
    row = [algorithm, n_flows, seed, k.throughput_gbps, k.reachability_frac,
           k.loss_rate_pct, k.mean_delay_ms, k.tail_delay_ms, k.jitter_ms,
           k.business_stability_frac, k.peak_util_frac, k.load_imbalance_frac,
           k.energy_kwh, k.lambda2]
    w = [f.offered_mbps for o in obs_list for f in o.flows]
    row.append(float(np.mean([o.mean_util for o in obs_list])))
    row.append(float(np.mean([o.top5pct_traffic_share for o in obs_list])))
    row.append(float(np.mean([o.overload_link_frac for o in obs_list])))
    row.append(float(np.mean([o.oversub_link_frac for o in obs_list])))
    row.append(_weighted_mean([f.hops for o in obs_list for f in o.flows], w))
    row.append(float(np.mean([o.n_active_edges for o in obs_list])))
    row.append(_weighted_mean([f.t_prop_ms for o in obs_list for f in o.flows], w))
    row.append(_weighted_mean([f.t_trans_ms for o in obs_list for f in o.flows], w))
    row.append(_weighted_mean([f.t_queue_ms for o in obs_list for f in o.flows], w))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--slots", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--flows", type=int, nargs="*", default=[500, 1000, 2000, 3000, 5000])
    ap.add_argument("--algorithms", nargs="*", default=ALGORITHMS)
    ap.add_argument("--out", default="results/raw_kpi_observations.csv")
    args = ap.parse_args()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                done.add((row["algorithm"], int(row["flows"]), int(row["seed"])))
        print(f"resume: {len(done)} rows already present in {out.name}", flush=True)

    # heaviest load first so the headline block (5,000 flows, all algorithms) lands early
    flows_desc = sorted(args.flows, reverse=True)
    jobs = [j for j in itertools.product(args.algorithms, flows_desc,
                                         range(1, args.seeds + 1), [args.slots])
            if (j[0], j[1], j[2]) not in done]
    print(f"{len(jobs)} runs to do; jobs={args.jobs}", flush=True)
    if not jobs:
        print("nothing to do", flush=True)
        return 0

    fh = out.open("a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if not done:
        w.writerow(FIELDS_ALL)
        fh.flush()
    t0 = time.time()
    n = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_worker, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            row = fut.result()
            w.writerow(row)
            fh.flush()
            n += 1
            el = time.time() - t0
            print(f"  {n}/{len(jobs)} [{el:6.0f}s] {j[0]} flows={j[1]} seed={j[2]}",
                  flush=True)
    fh.close()
    print(f"done: {n} new rows -> {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
