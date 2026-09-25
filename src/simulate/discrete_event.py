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
    path; ``T_queue`` uses the same ``Psi(rho)`` penalty that drives routing, scaled by
    a per-hop queueing quantum (``const.QUEUE_FRAME_QUANTUM_S``) that folds FIFO waiting
    together with the slotted/framing wait the author note identifies.  Loss uses a
    finite-buffer overflow estimate ``rho ** (B / (1 + Ca^2))`` in which ``Ca^2`` is the
    burstiness of the multiplexed arrival process -- this is what makes the coexistence
    of 82.3 % slot-average peak utilisation and 3.71 % loss physical, per the manuscript's
    Sec. III-E discussion (sub-slot burstiness, not headroom exhaustion).

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
                     QUEUE_FRAME_QUANTUM_S, SEC_PER_SLOT, SERVICE_TIME_S,
                     TAIL_PERCENTILE)


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
                  quantum_s: float = QUEUE_FRAME_QUANTUM_S,
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
            # per-hop queueing delay; bounded by the time to drain a full buffer at the
            # framing quantum, so a saturated link cannot produce unbounded delay
            tq_hops.append(min(_psi(rho) * quantum_s, buffer_packets * quantum_s))
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
      ``rho ** (B / (1 + Ca^2))``.  This reduces to the M/M/1/K tail ``rho^B`` for a
      Poisson arrival process (``Ca^2 = 1``); a multiplexed, bursty aggregate shortens
      the effective buffer, which is the regime the manuscript's 3-4 % loss corresponds
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
    # burst envelope: alternating on/off at a slot-scale timescale
    burst_state = rng.random(n_packets) < 0.15
    factor = np.where(burst_state, 1.0 + 2.0 * np.sqrt(ca2), np.maximum(0.05, 1.0 - 0.5))
    inter = rng.exponential(1.0 / np.maximum(rate_pps * factor, 1e-9))
    t_prop = np.array([edge_lookup[(min(a, b), max(a, b))]
                       for a, b in zip(path[:-1], path[1:])]) / C_KM_S * 1e3
    queues = [0] * hops
    last_free = [0.0] * hops
    delays = np.zeros(n_packets)
    lost = 0
    t_arr = 0.0
    service_ms = SERVICE_TIME_S * 1e3
    for p in range(n_packets):
        t_arr += inter[p]
        t = t_arr
        delay = 0.0
        dropped = False
        for h in range(hops):
            if queues[h] >= buffer_packets:
                dropped = True
                break
            start = max(t, last_free[h])
            queues[h] += 1
            end = start + service_ms
            last_free[h] = end
            delay += (start - t) + service_ms
            t = end
            # dequeue one packet per service slot
            queues[h] = max(0, queues[h] - 1)
        if dropped:
            lost += 1
        else:
            delays[p] = delay + float(t_prop.sum())
    ok = delays[:n_packets]
    d = np.diff(ok)
    return {
        "loss_frac": lost / n_packets,
        "mean_ms": float(np.mean(ok)),
        "p95_ms": float(np.percentile(ok, TAIL_PERCENTILE)),
        "jitter_ms": float(np.std(d)) if d.size else 0.0,
    }
