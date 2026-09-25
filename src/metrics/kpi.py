# -*- coding: utf-8 -*-
"""KPI aggregation and the 11-KPI composite score.

Manuscript Sec. IV-C, Eq. (composite).  Eleven KPIs are min-max normalised across the
compared algorithms *within each experimental configuration*, with the direction
flipped for metrics where lower is better, then combined with the Delphi weights
(0.5-1.4, total 10.6).  The composite is an evaluation metric only; it never enters the
solver.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# name -> (weight, higher_is_better, unit)
KPI_DEFS = {
    "throughput":        (1.4, True,  "Gbps"),
    "reachability":      (1.2, True,  "%"),
    "loss_rate":         (1.3, False, "%"),
    "mean_delay":        (0.9, False, "s"),
    "tail_delay":        (0.8, False, "s"),
    "jitter":            (1.0, False, "s"),
    "business_stability": (0.8, True, "%"),
    "peak_util":         (1.0, False, "%"),
    "load_imbalance":    (1.0, False, "-"),
    "energy":            (0.7, False, "kWh"),
    "lambda2":           (0.5, True,  "-"),
}
WEIGHT_SUM = sum(w for w, _, _ in KPI_DEFS.values())   # 10.6

# raw observation names used by the correlation study (10_kpi_raw_observations.csv)
RAW_KPI_ORDER = ["throughput_Gbps", "reachability_frac", "loss_rate_pct", "mean_delay_ms",
                 "tail_delay_ms", "jitter_ms", "business_stability_frac", "peak_util_frac",
                 "load_imbalance_frac", "energy_J_per_packet", "lambda2"]


@dataclass
class ConfigKPIs:
    """Per-(algorithm, flow-count, seed) KPI vector in SI-friendly units."""

    throughput_gbps: float
    reachability_frac: float
    loss_rate_pct: float
    mean_delay_ms: float
    tail_delay_ms: float
    jitter_ms: float
    business_stability_frac: float
    peak_util_frac: float
    load_imbalance_frac: float
    energy_kwh: float
    lambda2: float

    def as_named(self) -> dict:
        return {
            "throughput": self.throughput_gbps,
            "reachability": 100.0 * self.reachability_frac,
            "loss_rate": self.loss_rate_pct,
            "mean_delay": self.mean_delay_ms / 1e3,
            "tail_delay": self.tail_delay_ms / 1e3,
            "jitter": self.jitter_ms / 1e3,
            "business_stability": 100.0 * self.business_stability_frac,
            "peak_util": 100.0 * self.peak_util_frac,
            "load_imbalance": self.load_imbalance_frac,
            "energy": self.energy_kwh,
            "lambda2": self.lambda2,
        }


def observations_to_kpis(obs_list) -> ConfigKPIs:
    """Aggregate a run's per-slot observations into one KPI vector."""
    if not obs_list:
        return ConfigKPIs(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    goodput = np.mean([o.goodput_mbps for o in obs_list])
    reach = np.mean([o.reachability for o in obs_list])
    loss = np.mean([o.loss_rate for o in obs_list]) * 100.0
    mdel = np.mean([o.mean_delay_ms for o in obs_list])
    tdel = np.mean([o.tail_delay_ms for o in obs_list])
    jit = np.mean([o.jitter_ms for o in obs_list])
    bstab = np.mean([o.business_stability for o in obs_list])
    peak = np.mean([o.peak_util for o in obs_list])
    imb = np.mean([o.load_imbalance for o in obs_list])
    energy = np.mean([o.energy_kwh for o in obs_list])
    lam2 = np.mean([o.lambda2 for o in obs_list])
    return ConfigKPIs(goodput / 1e3, reach, loss, mdel, tdel, jit, bstab, peak, imb,
                      energy, lam2)


def composite_scores(kpi_by_algorithm: dict[str, ConfigKPIs]) -> dict[str, float]:
    """Min-max composite (Eq. composite) over the algorithms present in one config."""
    if not kpi_by_algorithm:
        return {}
    names = list(KPI_DEFS)
    table = {a: k.as_named() for a, k in kpi_by_algorithm.items()}
    algs = list(table)
    out = {a: 0.0 for a in algs}
    for name in names:
        w, higher_better, _ = KPI_DEFS[name]
        vals = np.array([table[a][name] for a in algs], dtype=float)
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        if hi - lo < 1e-12:
            norm = np.ones_like(vals)
        else:
            norm = (vals - lo) / (hi - lo)
            if not higher_better:
                norm = 1.0 - norm
        for a, nv in zip(algs, norm):
            out[a] += w * nv
    return {a: out[a] / WEIGHT_SUM for a in algs}


def ranking(kpi_by_algorithm: dict[str, ConfigKPIs]) -> list[tuple[str, float]]:
    sc = composite_scores(kpi_by_algorithm)
    return sorted(sc.items(), key=lambda kv: -kv[1])


def pairwise_correlation(rows: list[ConfigKPIs]) -> np.ndarray:
    """Correlation matrix over raw per-run KPI observations (Table 7 in the paper)."""
    mat = np.array([[r.throughput_gbps, r.reachability_frac, r.loss_rate_pct,
                     r.mean_delay_ms, r.tail_delay_ms, r.jitter_ms,
                     r.business_stability_frac, r.peak_util_frac,
                     r.load_imbalance_frac, r.energy_kwh, r.lambda2] for r in rows])
    return np.corrcoef(mat, rowvar=False)


def bandwidth_delay_efficiency(throughput_gbps: float, mean_delay_s: float) -> float:
    """B-D efficiency [Gbps/s] = delivered throughput / mean delay."""
    return throughput_gbps / mean_delay_s if mean_delay_s > 0 else float("nan")


def to_dict(k: ConfigKPIs) -> dict:
    return asdict(k)
