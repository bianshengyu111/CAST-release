# -*- coding: utf-8 -*-
"""Packet-level queueing evaluation.

The manuscript's loss / jitter / tail-delay figures come from a discrete-event engine
(StarPerf 2.0 in the original study) with:

* one FIFO output queue per ISL direction, no priority classes, no WFQ;
* fixed 1,500-B (12,000-bit) packets;
* a finite 100-packet buffer, tail-drop on overflow, **no ARQ and no retransmission**;
* delay decomposition T_prop + T_trans + T_queue, backbone only.

This module offers two interchangeable back-ends:

``analytic`` (default)
    A closed-form queueing evaluation.  ``T_prop`` and ``T_trans`` are exact given the
    path; per-hop ``T_queue`` is the M/M/1-form waiting time ``Psi(rho) * t_s`` capped
    by the time to drain a full buffer, ``B * t_s = 58.6 us``, with ``t_s = 0.586 us``
    the service time stated in the manuscript Sec. III-E -- so a saturated link cannot
    produce unbounded delay and path-total queueing is bounded by
    ``n_hops * 58.6 us``.  Loss uses a finite-buffer overflow estimate
    ``rho ** (B / (1 + Ca^2))`` in which ``Ca^2`` is the burstiness of the multiplexed
    arrival process (the manuscript's Sec. III-E sub-slot burstiness discussion).

``packet``
    A genuine per-packet FIFO finite-buffer simulation.  Faithful but expensive; use it
    on small constellations or single links to validate the analytic model, not for the
    full 1,401-satellite sweep (where the packet count is ~10^8 per slot).

Both back-ends return the same record schema, so metric aggregation is shared.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import norm

from ..const import (BUFFER_PACKETS, C_KM_S, ISL_CAPACITY_MBPS, PACKET_BITS,
                     SEC_PER_SLOT, SERVICE_TIME_S, TAIL_PERCENTILE)


def _psi(rho: float, eps: float = 1e-6) -> float:
    return rho / (1.0 - rho + eps)


@dataclass
class FlowRecord:
    src: int
    dst: int
    offered_mbps: float
    delivered_mbps: float
    loss_frac: float
    hops: int
    t_prop_ms: float
    t_trans_ms: float
    t_queue_ms: float
    mean_delay_ms: float
    tail_delay_ms: float
    jitter_ms: float


@dataclass
class SlotObservation:
    flows: list[FlowRecord] = field(default_factory=list)
    mean_util: float = 0.0
    peak_util: float = 0.0
    load_imbalance: float = 0.0
    top5pct_traffic_share: float = 0.0
    overload_link_frac: float = 0.0
    oversub_link_frac: float = 0.0
    energy_kwh: float = 0.0
    n_active_edges: int = 0
    lambda2: float = 0.0
    loss_rate: float = 0.0
    goodput_mbps: float = 0.0
    business_stability: float = 0.0
    reachability: float = 0.0
    n_flows: int = 0


def evaluate_slot(paths: Sequence[Sequence[int]],
                  commodities: Sequence[tuple[int, int, float]],
                  edge_lookup: dict[tuple[int, int], float],
                  utilization: dict[tuple[int, int], float],
                  n_sat: int,
                  ca2: float = 4.0,
                  buffer_packets: int = BUFFER_PACKETS,
                  service_time_s: float = SERVICE_TIME_S,
                  window_s: float = 60 * SEC_PER_SLOT,
                  kappa: float = 0.05,
                  overload_threshold: float = 0.75,
                  sla_ms: float = 400.0) -> SlotObservation:
    """Analytic per-slot evaluation of a routed topology."""
    obs = SlotObservation()
    if not paths:
        obs.n_active_edges = len(edge_lookup)
        return obs

    z = norm.ppf(TAIL_PERCENTILE / 100.0)
    total_offered = total_delivered = 0.0
    total_lost = 0.0
    tails, jitters, means = [], [], []
    weights = []

    for path, (s, d, b) in zip(paths, commodities):
        hops = len(path) - 1
        t_prop = sum(edge_lookup.get((min(a, c), max(a, c)), 0.0)
                     for a, c in zip(path[:-1], path[1:])) / C_KM_S * 1e3
        t_trans = hops * SERVICE_TIME_S * 1e3
        tq_hops = []
        p_survive = 1.0
        for a, c in zip(path[:-1], path[1:]):
            k = (min(a, c), max(a, c))
            rho = min(utilization.get(k, 0.0), 0.9999)
            # per-hop queueing delay: M/M/1-form waiting Psi(rho) * t_s, bounded by the
            # time to drain a full buffer B * t_s = 58.6 us, so a saturated link cannot
            # produce unbounded delay (Sec. III-E queueing model).  The utilisation is
            # capped at the model's ceiling RHO_EFF_MAX: the effective-buffer model is
            # defined on the sub-saturation branch, and a link at or above saturation is
            # charged that ceiling value (a conservative over-estimate of its loss).
            tq_hops.append(min(_psi(rho) * service_time_s,
                               buffer_packets * service_time_s))
            p_survive *= (1.0 - _link_loss(rho, buffer_packets, ca2))
        t_queue = sum(tq_hops) * 1e3
        mean_ms = t_prop + t_trans + t_queue
        # path variability: per-hop exponential queueing + routing-path dispersion
        sigma_ms = np.sqrt(sum((v * 1e3) ** 2 for v in tq_hops)) if tq_hops else 0.0
        loss = 1.0 - p_survive
        delivered = b * p_survive
        tail_ms = mean_ms + z * sigma_ms
        jitter_ms = np.sqrt(2.0) * sigma_ms

        obs.flows.append(FlowRecord(s, d, b, delivered, loss, hops, t_prop, t_trans,
                                    t_queue, mean_ms, tail_ms, jitter_ms))
        total_offered += b
        total_delivered += delivered
        total_lost += b * loss
        means.append(mean_ms)
        tails.append(tail_ms)
        jitters.append(jitter_ms)
        weights.append(b)

    w = np.asarray(weights)
    obs.mean_delay_ms = float(np.average(means, weights=w)) if w.sum() else 0.0
    obs.tail_delay_ms = float(np.average(tails, weights=w)) if w.sum() else 0.0
    obs.jitter_ms = float(np.average(jitters, weights=w)) if w.sum() else 0.0
    obs.loss_rate = total_lost / total_offered if total_offered else 0.0
    obs.goodput_mbps = total_delivered

    rhos_raw = np.array(list(utilization.values())) if utilization else np.zeros(1)
    # a link cannot physically exceed its capacity: excess load is dropped (see
    # _link_loss), so the utilisation statistics are capped at 1 and the oversubscribed
    # links are reported separately.
    rhos = np.minimum(rhos_raw, 1.0)
    obs.mean_util = float(rhos.mean())
    obs.peak_util = float(rhos.max())
    obs.oversub_link_frac = float((rhos_raw > 1.0).mean())
    obs.load_imbalance = float(rhos.std())
    if rhos.sum() > 0:
        top = np.sort(rhos)[::-1]
        k = max(1, int(np.ceil(0.05 * len(rhos))))
        obs.top5pct_traffic_share = float(top[:k].sum() / rhos.sum())
    obs.overload_link_frac = float((rhos > overload_threshold).mean())
    obs.n_active_edges = len(edge_lookup)
    # energy: kappa [W/km] * length [km] summed over active links, integrated over window
    total_w = sum(kappa * dist for dist in edge_lookup.values())
    obs.energy_kwh = total_w * window_s / 3.6e6
    obs.business_stability = float(np.mean([1.0 if t <= sla_ms else 0.0 for t in tails]))
    obs.reachability = 1.0
    obs.n_flows = len(obs.flows)
    return obs


def _link_loss(rho: float, buffer_packets: int, ca2: float) -> float:
    """Fraction of offered load a link fails to deliver.

    Two regimes:

    * ``rho < 1`` -- finite-buffer overflow with burstiness ``Ca^2``:
      ``rho ** (B / (1 + Ca^2))``.  A smooth aggregate (``Ca^2 = 0``) keeps the full
      buffer ``B`` in the exponent; a multiplexed, bursty aggregate shortens the
      effective buffer, which is the regime the manuscript's loss figures correspond
      to (Sec. III-E).
    * ``rho >= 1`` -- the link is fluid-saturated: it can only carry ``C``, so the
      excess ``lambda - C`` is dropped, i.e. a loss fraction of ``1 - 1/rho``.
    """
    if rho <= 0.0:
        return 0.0
    if rho >= 1.0:
        return float(1.0 - 1.0 / rho)
    k_eff = max(1.0, buffer_packets / (1.0 + ca2))
    return float(min(1.0, rho ** k_eff))


def packet_simulate_path(path: Sequence[int], edge_lookup: dict[tuple[int, int], float],
                         rate_mbps: float, n_packets: int, seed: int,
                         ca2: float = 4.0,
                         buffer_packets: int = BUFFER_PACKETS) -> dict:
    """Per-packet FIFO finite-buffer simulation along a fixed path.

    Offered packets are generated as a bursty (on/off-modulated) Poisson stream to
    reproduce the sub-slot burstiness the analytic model approximates.  Returns packet
    count, loss, mean/p95 delay and jitter for one flow -- used to validate
    ``evaluate_slot`` on reduced instances.
    """
    rng = np.random.default_rng(seed)
    hops = len(path) - 1
    if hops <= 0 or n_packets <= 0:
        return {"loss_frac": 0.0, "mean_ms": 0.0, "p95_ms": 0.0, "jitter_ms": 0.0}
    rate_pps = rate_mbps * 1e6 / PACKET_BITS
    # Burst envelope: on/off modulation with E[factor] = 1, so the offered *mean* rate
    # is exactly ``rate_mbps`` whatever the burst parameters are.  ``ca2`` sets the
    # peak/off ratio; the resulting Ca^2 at the multiplexed link is reported by
    # scripts/validate_queue_model.py so the two back-ends are compared at a stated
    # burstiness rather than an assumed one.
    p_on = 0.15
    f_hi = 1.0 + 2.0 * np.sqrt(ca2)
    f_lo = (1.0 - p_on * f_hi) / (1.0 - p_on)
    f_lo = f_lo if f_lo > 0.01 else 0.01
    burst_state = rng.random(n_packets) < p_on
    factor = np.where(burst_state, f_hi, f_lo)
    inter_s = rng.exponential(1.0 / np.maximum(rate_pps * factor, 1e-9))
    # exponential() on a modulated rate makes the realised mean depend on E[1/factor],
    # so rescale to pin the realised offered load at exactly ``rate_mbps``; the burst
    # structure (and hence the realised Ca^2) is preserved.
    inter_s *= (1.0 / rate_pps) / float(np.mean(inter_s))
    inter_ms = inter_s * 1e3                       # the loop below works in ms
    t_prop = np.array([edge_lookup[(min(a, b), max(a, b))]
                       for a, b in zip(path[:-1], path[1:])]) / C_KM_S * 1e3
    # per-hop single-server FIFO with a finite ``buffer_packets`` tail-drop queue;
    # ``free_at[h]`` is the time the server at hop h becomes free, so the backlog a
    # packet finds there is (start - t) / t_s packets -- that is the quantity the
    # buffer limit applies to.
    free_at = [0.0] * hops
    delays = np.zeros(n_packets)
    alive = np.zeros(n_packets, dtype=bool)
    lost = 0
    t_arr = 0.0
    service_ms = SERVICE_TIME_S * 1e3
    for p in range(n_packets):
        t_arr += inter_ms[p]
        t = t_arr
        delay = 0.0
        dropped = False
        for h in range(hops):
            start = t if t > free_at[h] else free_at[h]
            queued = (start - t) / service_ms          # packets ahead, incl. in service
            if queued >= buffer_packets:
                dropped = True
                break
            free_at[h] = start + service_ms
            delay += (start - t) + service_ms
            t = free_at[h]
        if dropped:
            lost += 1
        else:
            delays[p] = delay + float(t_prop.sum())
            alive[p] = True
    ok = delays[alive]                                # delivered packets only
    d = np.diff(ok)
    return {
        "loss_frac": lost / n_packets,
        "mean_ms": float(np.mean(ok)) if ok.size else 0.0,
        "p95_ms": float(np.percentile(ok, TAIL_PERCENTILE)) if ok.size else 0.0,
        "jitter_ms": float(np.std(d)) if d.size else 0.0,
    }
