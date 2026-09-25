#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate ``results/raw_kpi_observations.csv`` into the manuscript's tables.

Reproduces (in Markdown and LaTeX):

* Table 4  -- Overall performance ranking at the heaviest swept load.
* Table 5  -- Extended metrics (load imbalance, reachability, business stability,
              peak utilisation, energy).
* Table 6  -- Congestion regime across offered load and across methods.

The composite score is recomputed from the raw per-run KPIs with the manuscript's
min-max scheme, so the ranking is derived, not hard-coded.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.metrics.kpi import (ConfigKPIs, KPI_DEFS, composite_scores,  # noqa: E402
                             bandwidth_delay_efficiency)

ORDER = ["CAST", "Wang-MADRL", "6-nearest", "DGL-JCR", "Triangle", "Grid+",
         "Nie-DTC-DPSO"]


def load_rows(path: Path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def as_kpi(row) -> ConfigKPIs:
    return ConfigKPIs(
        throughput_gbps=float(row["throughput_gbps"]),
        reachability_frac=float(row["reachability_frac"]),
        loss_rate_pct=float(row["loss_rate_pct"]),
        mean_delay_ms=float(row["mean_delay_ms"]),
        tail_delay_ms=float(row["tail_delay_ms"]),
        jitter_ms=float(row["jitter_ms"]),
        business_stability_frac=float(row["business_stability_frac"]),
        peak_util_frac=float(row["peak_util_frac"]),
        load_imbalance_frac=float(row["load_imbalance_frac"]),
        energy_kwh=float(row["energy_kwh"]),
        lambda2=float(row["lambda2"]),
    )


def mean_kpis(rows) -> ConfigKPIs:
    ks = [as_kpi(r) for r in rows]
    fields = list(ks[0].__dataclass_fields__)
    return ConfigKPIs(**{f: float(np.mean([getattr(k, f) for k in ks])) for f in fields})


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="results/raw_kpi_observations.csv")
    ap.add_argument("--outdir", default="results/tables")
    args = ap.parse_args()

    rows = load_rows(ROOT / args.raw)
    by_alg_flow = defaultdict(list)
    for r in rows:
        by_alg_flow[(r["algorithm"], int(r["flows"]))].append(r)

    flows = sorted({int(r["flows"]) for r in rows})
    heaviest = max(flows)
    algs = [a for a in ORDER if any(k[0] == a for k in by_alg_flow)]
    algs += [a for a in {k[0] for k in by_alg_flow} if a not in algs]

    # ---- Table 4: headline at the heaviest load ------------------------------
    kpi_heavy = {a: mean_kpis(by_alg_flow[(a, heaviest)]) for a in algs}
    comp = composite_scores(kpi_heavy)
    rank = sorted(algs, key=lambda a: -comp.get(a, 0.0))
    t4 = []
    for a in rank:
        k = kpi_heavy[a].as_named()
        t4.append([a, f"{comp[a]:.3f}", f"{k['throughput'] * 1e3:,.1f}",
                   f"{k['loss_rate']:.2f}", f"{k['mean_delay']:.3f}",
                   f"{k['tail_delay']:.3f}", f"{k['jitter']:.3f}"])

    # ---- Table 5: extended metrics ------------------------------------------
    t5 = []
    for a in rank:
        k = kpi_heavy[a].as_named()
        bde = bandwidth_delay_efficiency(k["throughput"], k["mean_delay"])
        t5.append([a, f"{k['load_imbalance']:.4f}", f"{bde:,.0f}",
                   f"{k['reachability']:.1f}", f"{k['business_stability']:.1f}",
                   f"{k['peak_util']:.1f}", f"{k['energy']:.1f}"])

    # ---- Table 6: congestion across offered load (reference algorithm) -------
    ref = "CAST"
    t6 = []
    for f in flows:
        if (ref, f) not in by_alg_flow:
            continue
        k = mean_kpis(by_alg_flow[(ref, f)]).as_named()
        t6.append([f"{f:,} flows", f"{k['peak_util']:.1f}", f"{k['loss_rate']:.2f}",
                   f"{k['load_imbalance']:.4f}"])

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    body = []
    body.append(f"# Reproduced tables ({args.raw})\n")
    body.append(f"Heaviest swept load: **{heaviest:,} flows**; "
                f"{len(by_alg_flow[(algs[0], heaviest)])} seeds per cell.\n")
    body.append("## Table 4 -- Overall performance ranking\n")
    body.append(md_table(["Algorithm", "Composite", "Goodput (Mbps)", "Loss (%)",
                          "Mean delay (s)", "Tail delay (s)", "Jitter (s)"], t4))
    body.append("\n## Table 5 -- Extended metrics\n")
    body.append(md_table(["Algorithm", "Load imbalance", "B-D eff. (Gbps/s)",
                          "Reach. (%)", "Bus. stab. (%)", "Peak util. (%)",
                          "Energy (kWh)"], t5))
    body.append(f"\n## Table 6 -- Congestion regime ({ref} across offered load)\n")
    body.append(md_table(["Offered load", "Peak util. (%)", "Loss (%)",
                          "Load imbalance"], t6))
    body.append("\n## KPI weights\n")
    body.append(md_table(["KPI", "Weight", "Higher is better"],
                         [[n, w, h] for n, (w, h, _u) in KPI_DEFS.items()]))
    (outdir / "reproduced_tables.md").write_text("\n".join(body), encoding="utf-8")
    print(f"wrote {(outdir / 'reproduced_tables.md').relative_to(ROOT)}")

    # machine-readable companion
    comp_out = outdir / "composite_scores.csv"
    with comp_out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["algorithm", "composite", "rank"])
        for i, a in enumerate(rank, 1):
            w.writerow([a, f"{comp[a]:.6f}", i])
    print(f"wrote {comp_out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
