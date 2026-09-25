# -*- coding: utf-8 -*-
"""Congestion-aware sequential routing (manuscript Sec. III-D).

Flows are processed in descending bandwidth order and the congestion penalty is
updated as demand is assigned, so a later flow sees the residual capacity its
predecessors left behind.  This is what prevents several flows from independently
picking the same high-capacity path and only later discovering, at packet level, that
it is oversubscribed.

Two implementation choices keep the reference implementation tractable at N = 1,401:

* **Commodity aggregation.** Model flows are aggregated into ``D_sr(t)`` (Eq. 9) and
  each distinct (source satellite, destination satellite) pair is routed once.  With
  28 cities there are at most 756 such pairs, versus up to 5,000 flows.
* **Source-batched Dijkstra.** Within a weight snapshot, commodities are grouped by
  source satellite and one batched multi-source Dijkstra call resolves all of them,
  instead of one call per commodity.  ``batch_size`` sets how many commodities share a
  weight snapshot: ``batch_size=1`` is the strict per-commodity sequential update of
  Algorithm 1, while larger values trade a little concurrency fidelity for a large
  speed-up (the default 64 keeps the full matrix inside the runtime stated in the
  README).

Edge weight (Eq. 2, with the range normalisation the manuscript's code applies):

    w_ij = alpha_d * (d_ij / d_max) + alpha_c * Psi(rho_ij)
    Psi(rho) = rho / (1 - rho + eps)                      (Eq. 10)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from ..const import (ALPHA_C, ALPHA_D, EPS_RHO, ISL_CAPACITY_MBPS,
                     ISL_MAX_RANGE_KM)


@dataclass
class RoutingResult:
    loads_mbps: dict[tuple[int, int], float] = field(default_factory=dict)
    utilization: dict[tuple[int, int], float] = field(default_factory=dict)
    paths: list[list[int]] = field(default_factory=list)          # node sequences
    commodities: list[tuple[int, int, float]] = field(default_factory=list)
    unrouted: int = 0


def _psi(rho: np.ndarray, eps: float = EPS_RHO) -> np.ndarray:
    """Congestion penalty Psi(rho) = rho / (1 - rho + eps), Eq. (10).

    ``rho`` is clipped below 1 so the penalty stays finite and non-negative even when a
    link is momentarily oversubscribed during the sequential residual pass (an
    unclipped rho > 1 would make the Dijkstra weights negative).
    """
    rho = np.clip(np.asarray(rho, dtype=float), 0.0, 1.0 - 1e-3)
    return rho / (1.0 - rho + eps)


def route_commodities(n_sat: int,
                      edge_index: np.ndarray,
                      edge_dist: np.ndarray,
                      commodities: Sequence[tuple[int, int, float]],
                      capacity_mbps: float = ISL_CAPACITY_MBPS,
                      alpha_d: float = ALPHA_D,
                      alpha_c: float = ALPHA_C,
                      d_max: float = ISL_MAX_RANGE_KM,
                      batch_size: int = 64) -> RoutingResult:
    """Sequential residual-capacity routing with source-batched Dijkstra."""
    n_sat = int(n_sat)
    res = RoutingResult()
    if edge_index.size == 0:
        res.unrouted = len(commodities)
        return res

    m = edge_index.shape[0]
    didx = np.concatenate([np.arange(m), np.arange(m)])
    rows = np.concatenate([edge_index[:, 0], edge_index[:, 1]])
    cols = np.concatenate([edge_index[:, 1], edge_index[:, 0]])
    dd = edge_dist[didx]
    order = np.lexsort((cols, rows))
    rows, cols, dd, didx = rows[order], cols[order], dd[order], didx[order]
    base = alpha_d * (dd / d_max)

    load = np.zeros(m)
    local: dict[tuple[int, int], int] = {}

    def key(i: int, j: int) -> tuple[int, int]:
        return (i, j) if i < j else (j, i)

    for e in range(m):
        local[key(int(edge_index[e, 0]), int(edge_index[e, 1]))] = e

    ordered = sorted((c for c in commodities if int(c[0]) != int(c[1]) and c[2] > 0),
                     key=lambda c: -c[2])
    batch = max(1, int(batch_size))
    for start in range(0, len(ordered), batch):
        chunk = ordered[start:start + batch]
        rho = load / capacity_mbps
        w = base + alpha_c * _psi(rho)[didx]
        graph = csr_matrix((w, (rows, cols)), shape=(n_sat, n_sat))

        sources = sorted({int(c[0]) for c in chunk})
        src_row = {s: r for r, s in enumerate(sources)}
        dist, pred = dijkstra(graph, directed=True, indices=np.array(sources),
                              return_predecessors=True)
        dist = np.atleast_2d(dist)
        pred = np.atleast_2d(pred)

        for s, d, b in chunk:
            s, d = int(s), int(d)
            r = src_row[s]
            if not np.isfinite(dist[r, d]):
                res.unrouted += 1
                continue
            path = [d]
            cur = d
            while cur != s and pred[r, cur] >= 0:
                cur = int(pred[r, cur])
                path.append(cur)
            path.reverse()
            for a, c in zip(path[:-1], path[1:]):
                e = local.get(key(a, c))
                if e is not None:
                    load[e] += b
            res.paths.append(path)
            res.commodities.append((s, d, float(b)))

    res.loads_mbps = {key(int(edge_index[e, 0]), int(edge_index[e, 1])): float(load[e])
                      for e in range(m)}
    res.utilization = {k: v / capacity_mbps for k, v in res.loads_mbps.items()}
    return res


def total_queue_delay_ms(paths: Sequence[Sequence[int]],
                         util_lookup: dict[tuple[int, int], float],
                         capacity_weight: Sequence[float] | None = None,
                         frame_quantum_s: float | None = None) -> float:
    """Mean per-flow queueing delay [ms] using the per-hop Psi penalty."""
    from ..const import QUEUE_FRAME_QUANTUM_S
    q = QUEUE_FRAME_QUANTUM_S if frame_quantum_s is None else frame_quantum_s
    vals = []
    for p in paths:
        acc = float(np.sum(_psi(np.array([util_lookup.get(
            (a, b) if a < b else (b, a), 0.0) for a, b in zip(p[:-1], p[1:])]))))
        vals.append(acc * q)
    if not vals:
        return 0.0
    if capacity_weight is None:
        return float(np.mean(vals)) * 1e3
    w = np.asarray(capacity_weight, dtype=float)
    return float(np.average(vals, weights=w)) * 1e3
