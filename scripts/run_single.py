#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run one (algorithm, flow count, seed) configuration and print the KPI summary.

Examples
--------
    python scripts/run_single.py --algorithm CAST --flows 5000 --seed 1
    python scripts/run_single.py --algorithm 6-nearest --flows 1000 --seed 3 --slots 10
    python scripts/run_single.py --small            # 200-satellite smoke run
    python scripts/run_single.py --dump-raw         # write per-slot per-flow records
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constellation.walker import walker_72x20          # noqa: E402
from src.metrics.kpi import WEIGHT_SUM                     # noqa: E402
from src.sim.runner import RunConfig, run_configuration    # noqa: E402
from src.topology.baselines import ALGORITHMS              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algorithm", default="CAST", choices=ALGORITHMS)
    ap.add_argument("--flows", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--slots", type=int, default=60)
    ap.add_argument("--ca2", type=float, default=4.0, help="arrival burstiness Ca^2")
    ap.add_argument("--kappa", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="commodities sharing one routing weight snapshot sequential update)")

    ap.add_argument("--regime", default="dynamic", choices=["dynamic", "static"])
    ap.add_argument("--small", action="store_true",
                    help="200-satellite, 20-satellite-per-plane smoke configuration")
    ap.add_argument("--dump-raw", action="store_true")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    if args.small:
        walker = walker_72x20(planes=10, sats_per_plane=20, active=200)
        planes, spp = 10, 20
        args.slots = min(args.slots, 5)
        args.flows = min(args.flows, 300)
    else:
        walker = walker_72x20()
        planes, spp = 72, 20

    cfg = RunConfig(algorithm=args.algorithm, n_flows=args.flows, seed=args.seed,
                    n_slots=args.slots, ca2=args.ca2, kappa=args.kappa,
                    regime=args.regime, routing_batch=args.batch_size)

    def progress(slot, obs):
        print(f"  slot {slot + 1:>2}/{args.slots}  edges={obs.n_active_edges:<6} "
              f"peak={obs.peak_util * 100:5.1f}%  loss={obs.loss_rate * 100:5.2f}%  "
              f"delay={obs.mean_delay_ms:6.2f} ms", flush=True)

    print(f"[{args.algorithm}] flows={args.flows} seed={args.seed} slots={args.slots} "
          f"satellites={walker.active}")
    kpis = run_configuration(cfg, walker, planes, spp, progress=progress)
    summary = kpis.as_named()
    print("\nKPI summary")
    for k, v in summary.items():
        print(f"  {k:<20} {v:,.4f}")
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"single_{args.algorithm.replace('+', 'plus')}_{args.flows}_{args.seed}.json"
    out.write_text(json.dumps({"config": vars(args), "kpis": summary,
                               "weight_sum": WEIGHT_SUM}, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
