# -*- coding: utf-8 -*-
"""CAST -- Congestion-Aware Spectral Topology (Algorithm 1).

Alternating optimisation.  Each iteration:

1. route all commodity flows on the current topology with congestion-aware shortest
   paths and sequential residual-capacity updates;
2. recompute link utilisation and the Fiedler vector;
3. score every inactive candidate edge with the unified four-term score Eq. (12);
4. add the highest-scoring edges, in descending order, while the degree constraint
   ``Delta_max`` is respected and the score exceeds the threshold ``delta``.

Static ISLs are not decision variables: they are carried across slots subject only to
the range re-check and they occupy degree at both endpoints.  Dynamic ISLs are
passively removed when they go out of range or lose all their flows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..const import (ALPHA_C, ALPHA_D, DELTA_SCORE, ETA, ISL_CAPACITY_MBPS,
                     ISL_MAX_RANGE_KM, MAX_DEGREE, MAX_ITER,
                     NODE_THRESHOLD_LANCZOS)
from ..routing.congestion_aware import route_commodities
from ..spectral.fiedler import fiedler_vector
from .visibility import candidate_edges


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


@dataclass
class TopologyState:
    """Active topology as an edge dict ``(i, j) -> length`` with adjacency degrees."""

    n_sat: int
    edges: dict[tuple[int, int], float] = field(default_factory=dict)

    @staticmethod
    def key(i: int, j: int) -> tuple[int, int]:
        return (i, j) if i < j else (j, i)

    def index_and_dist(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.edges:
            return np.zeros((0, 2), dtype=int), np.zeros(0)
        idx = np.array(list(self.edges.keys()), dtype=int)
        dst = np.array(list(self.edges.values()), dtype=float)
        return idx, dst

    def degree(self, i: int) -> int:
        return sum(1 for (a, b) in self.edges if a == i or b == i)

    def degrees(self) -> np.ndarray:
        deg = np.zeros(self.n_sat, dtype=int)
        for (a, b) in self.edges:
            deg[a] += 1
            deg[b] += 1
        return deg

    def add(self, i: int, j: int, d: float) -> None:
        self.edges[self.key(i, j)] = float(d)

    def remove(self, i: int, j: int) -> None:
        self.edges.pop(self.key(i, j), None)


def mst_cold_start(n_sat: int, cand: np.ndarray, d_max: float = ISL_MAX_RANGE_KM,
                   max_degree: int = MAX_DEGREE,
                   density_fill: bool = True, use_mst: bool = True) -> TopologyState:
    """Cold start for t = 0.

    Algorithm 1 initialises the first slot with an MST over the candidate set and then
    greedily adds edges in ascending distance order while the degree budget lasts.  With
    ``use_mst=False`` the MST step is skipped and only the descending-distance fill runs
    -- the cold start used by the "without MST warm start" ablation row.
    """
    st = TopologyState(n_sat)
    if cand.size == 0:
        return st
    order = np.argsort(cand[:, 2])
    parent = list(range(n_sat))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    deg = np.zeros(n_sat, dtype=int)
    # Kruskal MST
    if use_mst:
        for e in order:
            i, j = int(cand[e, 0]), int(cand[e, 1])
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            if deg[i] >= max_degree or deg[j] >= max_degree:
                continue
            parent[ri] = rj
            st.add(i, j, cand[e, 2])
            deg[i] += 1
            deg[j] += 1
    if density_fill:
        for e in order:
            i, j = int(cand[e, 0]), int(cand[e, 1])
            if (i, j) in st.edges or (j, i) in st.edges:
                continue
            if deg[i] >= max_degree or deg[j] >= max_degree:
                continue
            st.add(i, j, cand[e, 2])
            deg[i] += 1
            deg[j] += 1
    return st


def score_candidates(cand_pool: np.ndarray, state: TopologyState,
                     demand: np.ndarray, util_lookup: dict[tuple[int, int], float],
                     q: np.ndarray, eta: tuple[float, ...] = ETA,
                     d_max: float = ISL_MAX_RANGE_KM) -> np.ndarray:
    """Unified edge score Eq. (12) for every candidate edge in ``cand_pool``.

    Returns an ``(M,)`` array aligned with ``cand_pool``.  Each of the four terms is
    min-max normalised across the pool so that the ``eta_i`` are comparable on a
    unit-sum simplex (manuscript Sec. III-G).
    """
    if cand_pool.size == 0:
        return np.zeros(0)
    i = cand_pool[:, 0].astype(int)
    j = cand_pool[:, 1].astype(int)
    d = cand_pool[:, 2].astype(float)

    # --- traffic gain Eq. (traffic_gain): incident demand at both endpoints, both roles
    src = demand.sum(axis=1)     # sum_r D[i, r]
    dst = demand.sum(axis=0)     # sum_d D[d, j]
    g_tr = src[i] + dst[j] + src[j] + dst[i]

    # --- load-balancing gain Eq. (lb_gain): max utilisation at each endpoint
    max_util = np.zeros(state.n_sat)
    for (a, b), rho in util_lookup.items():
        max_util[a] = max(max_util[a], rho)
        max_util[b] = max(max_util[b], rho)
    g_lb = max_util[i] + max_util[j]

    # --- spectral gain Eq. (11): (q_i - q_j)^2
    g_sp = (q[i] - q[j]) ** 2

    # --- distance cost Eq. (dist_cost)
    c_dist = d / d_max

    eta1, eta2, eta3, eta4 = eta
    s = (eta1 * _minmax(g_tr) + eta2 * _minmax(g_lb)
         + eta3 * _minmax(g_sp) - eta4 * _minmax(c_dist))
    return s


def select_cast(n_sat: int, positions_km: np.ndarray, demand: np.ndarray,
                prev_state: TopologyState | None = None,
                static_edges: dict[tuple[int, int], float] | None = None,
                commodities: list[tuple[int, int, float]] | None = None,
                alpha_d: float = ALPHA_D, alpha_c: float = ALPHA_C,
                eta: tuple[float, ...] = ETA, delta: float = DELTA_SCORE,
                max_iter: int = MAX_ITER, max_degree: int = MAX_DEGREE,
                d_max: float = ISL_MAX_RANGE_KM,
                mst_init: bool = True, routing_batch: int = 64,
                allow_swap: bool = True,
                max_add_per_iter: int = 64) -> tuple[TopologyState, dict]:
    """Run Algorithm 1 for one slot and return the topology plus a diagnostics dict.

    ``mst_init=False`` selects the random-spanning-tree cold start used by the
    "without MST warm start" ablation row.
    """
    cand = candidate_edges(positions_km, d_max)
    cand_map = {(min(int(a), int(b)), max(int(a), int(b))): float(c)
                for a, b, c in cand}

    st = TopologyState(n_sat)
    if static_edges:
        for (a, b), d in static_edges.items():
            if (a, b) in cand_map:
                st.add(a, b, cand_map[(a, b)])

    if prev_state is None or not prev_state.edges:
        st = mst_cold_start(n_sat, cand, d_max, max_degree, density_fill=True,
                            use_mst=mst_init)
        if static_edges:
            for (a, b), d in static_edges.items():
                if (a, b) in cand_map:
                    st.add(a, b, cand_map[(a, b)])
        warm_start = False
    else:
        warm_start = True
        # static ISLs carried over subject only to the range re-check; dynamic ISLs are
        # pruned at the end of the previous slot (see below)
        keep = {k: v for k, v in prev_state.edges.items() if k in cand_map}
        st.edges.update(keep)

    hit_max_iter = True
    added_history: list[int] = []
    for it in range(max_iter):
        idx, dist = st.index_and_dist()
        comms = commodities if commodities is not None else _demand_to_commodities(demand)
        res = route_commodities(n_sat, idx, dist, comms,
                                alpha_d=alpha_d, alpha_c=alpha_c, d_max=d_max,
                                batch_size=routing_batch)
        util_lookup = res.utilization
        q = fiedler_vector(n_sat, [(int(a), int(b)) for a, b in idx],
                           NODE_THRESHOLD_LANCZOS)

        active = set(st.edges.keys())
        pool_mask = np.array([((min(int(a), int(b)), max(int(a), int(b))) not in active)
                              for a, b in cand[:, :2]])
        pool = cand[pool_mask]
        scores = score_candidates(pool, st, demand, util_lookup, q, eta, d_max)

        deg = st.degrees()
        bridges = _find_bridges(n_sat, list(st.edges)) if allow_swap else set()
        added = 0
        if scores.size:
            for e in np.argsort(-scores):
                if scores[e] <= delta:
                    break
                a, b = int(pool[e, 0]), int(pool[e, 1])
                if (min(a, b), max(a, b)) in st.edges:
                    continue
                if allow_swap:
                    # degree budget exhausted: try a 1-swap, evicting the least-used
                    # incident edge, so that a high-scoring candidate can replace a
                    # geometry-only edge once the constellation is saturated
                    if deg[a] >= max_degree and not _evict_least_used(
                            st, a, deg, util_lookup, bridges):
                        continue
                    if deg[b] >= max_degree and not _evict_least_used(
                            st, b, deg, util_lookup, bridges):
                        continue
                if deg[a] >= max_degree or deg[b] >= max_degree:
                    continue
                st.add(a, b, pool[e, 2])
                deg[a] += 1
                deg[b] += 1
                added += 1
                if added >= max_add_per_iter:
                    break
        added_history.append(added)
        if added == 0:
            hit_max_iter = False
            break

    # final routing on the converged topology
    idx, dist = st.index_and_dist()
    comms = commodities if commodities is not None else _demand_to_commodities(demand)
    final = route_commodities(n_sat, idx, dist, comms,
                              alpha_d=alpha_d, alpha_c=alpha_c, d_max=d_max,
                              batch_size=routing_batch)

    # passive removal of idle dynamic ISLs: once a link no longer lies on any flow's
    # path it is dropped, which is what frees degree for the next slot's additions
    # (Algorithm 1, "passive removal").  Two safeguards keep the removal from hurting:
    # only non-bridge links are eligible (so the removed set is a subset of the
    # redundant links and connectivity is preserved with no repair step), and the
    # pruned topology is accepted only if it does not raise the peak link utilisation.
    # A strictly additive greedy neighbourhood cannot pass this test on its own once the
    # degree budget is saturated, so the acceptance test is what makes the passive step
    # safe rather than merely plausible.
    n_idle_removed = 0
    if warm_start and final.loads_mbps:
        idle = [k for k in st.edges if final.loads_mbps.get(k, 0.0) <= 0.0]
        if idle:
            bridges = _find_bridges(n_sat, list(st.edges))
            removable = [k for k in idle if k not in bridges]
            if removable:
                base_peak = max(final.utilization.values()) if final.utilization else 0.0
                snapshot = dict(st.edges)
                for k in removable:
                    st.edges.pop(k, None)
                idx, dist = st.index_and_dist()
                pruned = route_commodities(n_sat, idx, dist, comms,
                                           alpha_d=alpha_d, alpha_c=alpha_c, d_max=d_max,
                                           batch_size=routing_batch)
                new_peak = max(pruned.utilization.values()) if pruned.utilization else 0.0
                if new_peak <= base_peak + 1e-9:
                    final = pruned
                    n_idle_removed = len(removable)
                else:
                    st.edges = snapshot      # revert: the pruned topology is worse

    diag = {
        "n_edges": len(st.edges),
        "edges_added_per_iter": added_history,
        "idle_removed": n_idle_removed,
        "warm_start": warm_start,
        "routing": final,
        "converged": not hit_max_iter,
    }
    return st, diag


def _evict_least_used(st: TopologyState, node: int, deg: np.ndarray,
                      util_lookup: dict[tuple[int, int], float],
                      bridges: set[tuple[int, int]] | None = None) -> bool:
    """Drop the least-used non-bridge incident edge of ``node``; update ``deg`` in place.

    Returns ``True`` when an edge was evicted (so a degree slot is now free).
    """
    incident = [k for k in st.edges if (k[0] == node or k[1] == node)
                and (bridges is None or k not in bridges)]
    if not incident:
        return False
    victim = min(incident, key=lambda k: util_lookup.get(k, 0.0))
    st.edges.pop(victim, None)
    deg[victim[0]] -= 1
    deg[victim[1]] -= 1
    return True


def _find_bridges(n_sat: int, edges: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """Iterative Tarjan bridge detection; returns the bridge edge keys."""
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n_sat)]
    for idx, (a, b) in enumerate(edges):
        adj[a].append((b, idx))
        adj[b].append((a, idx))
    disc = [-1] * n_sat
    low = [0] * n_sat
    bridges: set[tuple[int, int]] = set()
    timer = 0
    for root in range(n_sat):
        if disc[root] != -1:
            continue
        stack = [(root, -1, iter(adj[root]))]
        disc[root] = low[root] = timer
        timer += 1
        while stack:
            u, pe, it = stack[-1]
            advanced = False
            for v, ei in it:
                if ei == pe:
                    continue
                if disc[v] == -1:
                    disc[v] = low[v] = timer
                    timer += 1
                    stack.append((v, ei, iter(adj[v])))
                    advanced = True
                    break
                low[u] = min(low[u], disc[v])
            if advanced:
                continue
            stack.pop()
            if stack:
                p = stack[-1][0]
                low[p] = min(low[p], low[u])
                if low[u] > disc[p] and pe != -1:
                    bridges.add(edges[pe])
    return bridges


def _demand_to_commodities(demand: np.ndarray) -> list[tuple[int, int, float]]:
    nz = np.argwhere(demand > 0)
    return [(int(s), int(d), float(demand[s, d])) for s, d in nz]
