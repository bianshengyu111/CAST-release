#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ablation study (manuscript Table 8).

Zeroes out one or two edge-score components at a time, plus the MST warm start, and
reports each variant's composite score against the same min-max anchors used by the
baselines (i.e. the full 11-KPI composite is recomputed over {variants} U {baselines}
in each configuration, so the deltas are comparable to the headline numbers).

    python scripts/run_ablation.py --jobs 4 --flows 5000 --seeds 10
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constellation.walker import walker_72x20            # noqa: E402
from src.metrics.kpi import KPI_DEFS, composite_scores       # noqa: E402
from src.sim.runner import RunConfig, run_configuration      # noqa: E402
from src.topology.baselines import ALGORITHMS                # noqa: E402

BASE_ETA = (0.30, 0.25, 0.25, 0.20)


def _variant(name: str) -> tuple[str, tuple, bool]:
    drop = {"Traffic": 0, "LoadBal": 1, "Spectral": 2, "DistanceCost": 3}
    if name == "CAST (full)":
        return name, BASE_ETA, True
    if name == "CAST w/o MST warm start":
        return name, BASE_ETA, False
    if name.startswith("CAST w/o "):
        body = name[len("CAST w/o "):]
        off = [drop[p] for p in body.split("+")]
        eta = list(BASE_ETA)
        for o in off:
            eta[o] = 0.0
        s = sum(eta)
        eta = tuple(e / s * sum(BASE_ETA) for e in eta) if s > 0 else tuple(eta)
        return name, eta, True
    raise ValueError(name)


VARIANTS = [
    "CAST (full)",
    "CAST w/o Traffic", "CAST w/o Spectral", "CAST w/o LoadBal",
    "CAST w/o DistanceCost", "CAST w/o MST warm start",
    "CAST w/o Traffic+Spectral", "CAST w/o Traffic+LoadBal",
    "CAST w/o Spectral+LoadBal", "CAST w/o Spectral+DistanceCost",
    "CAST w/o LoadBal+DistanceCost",
]


def _worker(job):
    key, algorithm, eta, mst, n_flows, seed, slots = job
    w = walker_72x20()
    cfg = RunConfig(algorithm=algorithm, n_flows=n_flows, seed=seed, n_slots=slots,
                    eta=eta, mst_init=mst)
    k = run_configuration(cfg, w, 72, 20)
    return key, seed, k


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--flows", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--slots", type=int, default=60)
    ap.add_argument("--out", default="results/tables/ablation.md")
    args = ap.parse_args()

    specs = [_variant(v) for v in VARIANTS]
    jobs = [(name, "CAST", eta, mst, args.flows, s, args.slots)
            for (name, eta, mst) in specs for s in range(1, args.seeds + 1)]
    # baselines provide the min-max anchors
    jobs += [(a, a, BASE_ETA, True, args.flows, s, args.slots)
             for a in ALGORITHMS if a != "CAST" for s in range(1, args.seeds + 1)]

    print(f"{len(jobs)} runs")
    collected: dict[tuple, list] = {}
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(_worker, j) for j in jobs]
            for n, fut in enumerate(as_completed(futs), 1):
                key, seed, k = fut.result()
                collected.setdefault(key, []).append(k)
                print(f"  {n}/{len(jobs)}", flush=True)
    else:
        for n, j in enumerate(jobs, 1):
            key, seed, k = _worker(j)
            collected.setdefault(key, []).append(k)
            print(f"  {n}/{len(jobs)}  {key}", flush=True)

    # average per key
    from src.metrics.kpi import ConfigKPIs
    mean_k = {}
    for key, ks in collected.items():
        fields = list(ks[0].__dataclass_fields__)
        mean_k[key] = ConfigKPIs(**{f: float(np.mean([getattr(k, f) for k in ks]))
                                    for f in fields})
    comp = composite_scores(mean_k)

    rows = []
    full = comp.get("CAST (full)", float("nan"))
    for v in VARIANTS:
        if v not in comp:
            continue
        delta = (comp[v] - full) / full * 100.0 if full else 0.0
        rows.append((v, comp[v], delta))
    bl = [(a, comp[a]) for a in ALGORITHMS if a != "CAST" and a in comp]

    out = Path(ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Ablation ({args.flows:,} flows, {args.seeds} seeds)\n",
             "| Variant | Composite | Delta vs full |", "|---|---|---|"]
    for v, c, d in rows:
        lines.append(f"| {v} | {c:.4f} | {d:+.1f}% |")
    lines.append("\n## Baseline anchors\n")
    lines.append("| Algorithm | Composite |")
    lines.append("|---|---|")
    for a, c in sorted(bl, key=lambda kv: -kv[1]):
        lines.append(f"| {a} | {c:.4f} |")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
