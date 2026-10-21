# -*- coding: utf-8 -*-
"""CAST -- Congestion-Aware Spectral Topology (Algorithm 1).

Alternating optimisation.  Each iteration:

1. route all commodity flows on the current topology with congestion-aware shortest
   paths and sequential residual-capacity updates;
2. recompute link utilisation and the Fiedler vector;
3. score every inactive candidate edge with the unified four-term score Eq. (12);
4. add the highest-scoring edges, in descending order, while the degree constraint
   ``Delta_max`` is respected and the score exceeds the threshold ``delta``.

Static ISLs (the intra-plane ring) are not decision variables: they are carried across
slots subject only to the range re-check, they occupy degree at both endpoints, and
they are never idle-removed or evicted by a swap.  Dynamic ISLs are passively removed
when they go out of range or lose all their flows; the removed set excludes bridges,
so connectivity is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..const import (ALPHA_C, ALPHA_D, DELTA_SCORE, ETA, ISL_CAPACITY_MBPS,
                     ISL_MAX_RANGE_KM, KAPPA_W_PER_KM, MAX_DEGREE, MAX_ITER,
                     NODE_THRESHOLD_LANCZOS)
from ..routing.congestion_aware import route_commodities
from ..simulate.discrete_event import _link_loss
from ..spectral.fiedler import algebraic_connectivity, fiedler_vector
from .visibility import candidate_edges

# Global objective weights omega_1..4 of Eq. (6) (manuscript Sec. III-B).
# Diagnostics only: they enter the convergence trace recorded in ``diag`` and are
# never read by the selection logic.
J_OMEGA = (0.30, 0.25, 0.25, 0.20)


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _objective_terms(n_sat: int, idx: np.ndarray, dist: np.ndarray,
                     util_lookup: dict[tuple[int, int], float]) -> list[float]:
    """Raw Eq. (6) sub-objective values for one routed iteration (diagnostics only).

    Reading of the four sub-objectives, manuscript Sec. III:

    * ``C_load`` -- congestion cost: mean per-link utilisation
      ``rho_bar = (1/|E|) sum_e rho_e`` of the network under the current routing;
    * ``F_LB``   -- Eq. (3): population std of the per-link utilisations;
    * ``F_RB``   -- Eq. (4): lambda_2 of the active topology, computed with the
      release's own ``algebraic_connectivity`` (the exact value the KPI reports;
      measured ~0.06-0.2 s per call at N = 1,401);
    * ``E_tot``  -- Eq. (5): raw power index ``kappa * sum_e d_e`` [W].

    Returns ``[J_raw, C_load, F_LB, lambda2, E_tot_W]`` with
    ``J_raw = omega . (C_load, F_LB, -F_RB, E_tot)`` in raw units.  Pure recording:
    reads its arguments, mutates nothing, calls no RNG.
    """
    rhos = np.asarray(list(util_lookup.values()), dtype=float)
    c_load = float(rhos.mean()) if rhos.size else 0.0
    f_lb = float(rhos.std()) if rhos.size else 0.0
    lam2 = (algebraic_connectivity(n_sat, [(int(a), int(b)) for a, b in idx])
            if idx.size else 0.0)
    e_tot_w = KAPPA_W_PER_KM * float(dist.sum()) if dist.size else 0.0
    w1, w2, w3, w4 = J_OMEGA
    j_raw = w1 * c_load + w2 * f_lb - w3 * lam2 + w4 * e_tot_w
    return [j_raw, c_load, f_lb, lam2, e_tot_w]


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
                   density_fill: bool = True, use_mst: bool = True,
                   static_edges: dict[tuple[int, int], float] | None = None) -> TopologyState:
    """Cold start for t = 0.

    Algorithm 1 initialises the first slot with an MST over the candidate set and then
    greedily adds edges in ascending distance order while the degree budget lasts.  With
    ``use_mst=False`` the MST step is skipped and only the descending-distance fill runs
    -- the cold start used by the "without MST warm start" ablation row.

    Static ISLs are seeded first and their degree is counted against ``max_degree``:
    the manuscript states that static ISLs "occupy degree at both endpoints", so a
    satellite carrying two ring terminals enters the fill with four degree slots left,
    never with six.
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
    # Static ring edges arrive with placeholder lengths (walker.py resolves them from
    # the slot's candidate set); seed them with the real chord lengths here.
    cand_map = {(min(int(a), int(b)), max(int(a), int(b))): float(c)
                for a, b, c in cand}
    for (a, b), d in (static_edges or {}).items():
        st.add(a, b, cand_map.get((a, b), d))
        deg[a] += 1
        deg[b] += 1
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
    static_keys = {TopologyState.key(int(a), int(b))
                   for (a, b) in (static_edges or {})}

    st = TopologyState(n_sat)
    if static_edges:
        for (a, b), d in static_edges.items():
            if (a, b) in cand_map:
                st.add(a, b, cand_map[(a, b)])

    if prev_state is None or not prev_state.edges:
        st = mst_cold_start(n_sat, cand, d_max, max_degree, density_fill=True,
                            use_mst=mst_init, static_edges=static_edges)
        warm_start = False
    else:
        warm_start = True
        # static ISLs carried over subject only to the range re-check; dynamic ISLs are
        # pruned at the end of the previous slot (see below)
        keep = {k: v for k, v in prev_state.edges.items() if k in cand_map}
        st.edges.update(keep)

    hit_max_iter = True
    added_history: list[int] = []
    j_history: list[list[float]] = []   # Eq. (6) trace, one entry per iteration
    for it in range(max_iter):
        idx, dist = st.index_and_dist()
        comms = commodities if commodities is not None else _demand_to_commodities(demand)
        res = route_commodities(n_sat, idx, dist, comms,
                                alpha_d=alpha_d, alpha_c=alpha_c, d_max=d_max,
                                batch_size=routing_batch)
        util_lookup = res.utilization
        q = fiedler_vector(n_sat, [(int(a), int(b)) for a, b in idx],
                           NODE_THRESHOLD_LANCZOS)

        # convergence instrumentation: Eq. (6) objective of the topology *entering*
        # this iteration, evaluated on the fresh routing above.  Pure recording.
        j_history.append(_objective_terms(n_sat, idx, dist, util_lookup))

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
                    # guard: if *both* endpoints are saturated, evicting at each end
                    # would drop two links to add one and shrink the active set, and
                    # a swap that only succeeds at one end would drop a link for
                    # nothing.  Rejecting that case keeps every swap atomic.
                    if deg[a] >= max_degree and deg[b] >= max_degree:
                        continue
                    # degree budget exhausted: try a 1-swap, evicting the least-used
                    # incident edge, so that a high-scoring candidate can replace a
                    # geometry-only edge once the constellation is saturated
                    if deg[a] >= max_degree and not _evict_least_used(
                            st, a, deg, util_lookup, bridges, static_keys):
                        continue
                    if deg[b] >= max_degree and not _evict_least_used(
                            st, b, deg, util_lookup, bridges, static_keys):
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
    # (Algorithm 1, "passive removal").  The removed set is restricted, as in the
    # algorithm: static ISLs are never eligible (they are not decision variables), and
    # every removal is individually guarded so the graph stays connected -- edges
    # whose removal would disconnect the topology are load-bearing and are kept.  The
    # guard is applied per edge because a set of edges that are each non-bridges can
    # still jointly disconnect the graph.  The pruned topology is finally accepted
    # only if it does not raise the peak link utilisation.
    n_idle_removed = 0
    if warm_start and final.loads_mbps:
        idle = [k for k in st.edges
                if k not in static_keys and final.loads_mbps.get(k, 0.0) <= 0.0]
        if idle:
            # Algorithm 1, passive removal: "removed set is a subset of redundant
            # (non-MST) links, so connectivity is preserved".  The MST of the active
            # topology is the connectivity skeleton, so MST edges are never idle-removed
            # no matter how the per-slot routing happens to fall; bridges are excluded
            # as well (a non-MST edge of a connected graph is already a non-bridge --
            # the bridge test is kept purely as a guard).
            mst = _mst_edge_set(n_sat, st.edges)
            bridges = _find_bridges(n_sat, list(st.edges))
            removable = [k for k in idle if k not in mst and k not in bridges]
            if removable:
                base_peak = max(final.utilization.values()) if final.utilization else 0.0
                snapshot = dict(st.edges)
                removed: list[tuple[int, int]] = []
                for k in removable:
                    st.edges.pop(k, None)
                    if _is_connected(n_sat, list(st.edges)):
                        removed.append(k)
                    else:
                        st.edges[k] = snapshot[k]   # load-bearing: keep it
                if removed:
                    idx, dist = st.index_and_dist()
                    pruned = route_commodities(n_sat, idx, dist, comms,
                                               alpha_d=alpha_d, alpha_c=alpha_c,
                                               d_max=d_max,
                                               batch_size=routing_batch)
                    new_peak = max(pruned.utilization.values()) if pruned.utilization else 0.0
                    base_deliv = _delivered_frac(final)
                    new_deliv = _delivered_frac(pruned)
                    if (new_peak <= base_peak + 1e-9
                            and new_deliv >= base_deliv - 1e-9):
                        final = pruned
                        n_idle_removed = len(removed)
                    else:
                        st.edges = snapshot      # revert: the pruned topology is worse

    # convergence instrumentation for the final accepted topology (post passive
    # removal): the same read-only Eq. (6) evaluation on the final routing
    f_idx, f_dist = st.index_and_dist()
    j_final = _objective_terms(n_sat, f_idx, f_dist, final.utilization)

    diag = {
        "n_edges": len(st.edges),
        "edges_added_per_iter": added_history,
        "J_per_iter": [row[0] for row in j_history],
        "J_terms_per_iter": j_history,   # [J_raw, C_load, F_LB, lambda2, E_tot_W]
        "J_final": j_final[0],
        "J_final_terms": j_final,
        "idle_removed": n_idle_removed,
        "warm_start": warm_start,
        "routing": final,
        "converged": not hit_max_iter,
    }
    return st, diag


def _evict_least_used(st: TopologyState, node: int, deg: np.ndarray,
                      util_lookup: dict[tuple[int, int], float],
                      bridges: set[tuple[int, int]] | None = None,
                      protected: set[tuple[int, int]] | None = None) -> bool:
    """Drop the least-used non-bridge incident edge of ``node``; update ``deg`` in place.

    Static ISLs (``protected``) and bridges are never evicted.  Returns ``True`` when
    an edge was evicted (so a degree slot is now free).
    """
    incident = [k for k in st.edges if (k[0] == node or k[1] == node)
                and (bridges is None or k not in bridges)
                and (protected is None or k not in protected)]
    if not incident:
        return False
    victim = min(incident, key=lambda k: util_lookup.get(k, 0.0))
    st.edges.pop(victim, None)
    deg[victim[0]] -= 1
    deg[victim[1]] -= 1
    return True


def _delivered_frac(res, buffer_packets: int = 100, ca2: float = 4.0) -> float:
    """Delivered fraction of the offered load under the finite-buffer loss model.

    Same per-link loss formula as the KPI evaluation, so the passive-removal
    acceptance test measures the quantity the reported loss is made of.
    """
    delivered = 0.0
    offered = 0.0
    for path, (_s, _d, b) in zip(res.paths, res.commodities):
        p = 1.0
        for a, c in zip(path[:-1], path[1:]):
            k = (a, c) if a < c else (c, a)
            rho = min(res.utilization.get(k, 0.0), 0.9999)
            p *= (1.0 - _link_loss(rho, buffer_packets, ca2))
        delivered += b * p
        offered += b
    return delivered / offered if offered else 1.0


def _is_connected(n_sat: int, edges: list[tuple[int, int]]) -> bool:
    """BFS reachability check: True iff the undirected graph spans all n_sat nodes."""
    if n_sat <= 1:
        return True
    adj: list[list[int]] = [[] for _ in range(n_sat)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen = [False] * n_sat
    seen[0] = True
    stack = [0]
    count = 1
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                count += 1
                stack.append(v)
    return count == n_sat


def _mst_edge_set(n_sat: int, edges: dict[tuple[int, int], float]) -> set[tuple[int, int]]:
    """Kruskal MST of the active edge set (weight = chord length), as a set of edge keys.

    Deterministic: ties broken by edge key order.
    """
    order = sorted(edges.items(), key=lambda kv: (kv[1], kv[0]))
    parent = list(range(n_sat))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    mst: set[tuple[int, int]] = set()
    for (a, b), _d in order:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        parent[ra] = rb
        mst.add((a, b))
        if len(mst) == n_sat - 1:
            break
    return mst


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
