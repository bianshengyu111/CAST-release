#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot the manuscript's figures from ``results/raw_kpi_observations.csv``.

* Fig. 3 -- total throughput and packet-loss rate versus registered flow count.
* Fig. 4 -- mean delay and tail delay versus registered flow count.
* Fig. 5 -- convergence of J(t) for CAST (single configuration).

Output: ``results/figures/*.png`` (matplotlib, Agg backend).
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

import matplotlib                                               # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

LABEL = {"CAST": "CAST (proposed)", "6-nearest": "6-nearest", "Grid+": "Grid+",
         "Nie-DTC-DPSO": "Nie-DTC-DPSO", "Triangle": "Triangle",
         "Wang-MADRL": "Wang-MADRL", "DGL-JCR": "DGL-JCR"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="results/raw_kpi_observations.csv")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()

    with (ROOT / args.raw).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    flows = sorted({int(r["flows"]) for r in rows})
    by = defaultdict(list)
    for r in rows:
        by[(r["algorithm"], int(r["flows"]))].append(r)

    algs = [a for a in LABEL if any(k[0] == a for k in by)]

    def series(alg, col, scale=1.0):
        xs, ys = [], []
        for f in flows:
            cell = by.get((alg, f))
            if cell:
                xs.append(f)
                ys.append(scale * np.mean([float(c[col]) for c in cell]))
        return xs, ys

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    panels = [("throughput_gbps", 1e3, "Total throughput (Mbps)", "fig3a_throughput.png"),
              ("loss_rate_pct", 1.0, "Packet loss rate (%)", "fig3b_loss.png"),
              ("mean_delay_ms", 1.0, "Mean delay (ms)", "fig4a_mean_delay.png"),
              ("tail_delay_ms", 1.0, "Tail delay (95th, ms)", "fig4b_tail_delay.png")]
    for col, scale, ylab, name in panels:
        fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=150)
        for a in algs:
            xs, ys = series(a, col, scale)
            if xs:
                ax.plot(xs, ys, marker="o", ms=4, lw=1.6, label=LABEL[a])
        ax.set_xlabel("Registered flow count")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(outdir / name)
        plt.close(fig)
        print(f"wrote {(outdir / name).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
