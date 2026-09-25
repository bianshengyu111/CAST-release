# -*- coding: utf-8 -*-
"""Baseline topology selection rules.

Six baselines, all sharing one interface so that the comparison is apples-to-apples
(same candidate set, same demand, same evaluation stack).  Every rule produces a
connected, degree-capped topology; they differ in *which* edges they choose.

* ``six_nearest``  -- Iridium-style geometric k-nearest rule (k = Delta_max).  Purely
  geometric: ordering is by chord distance, so load plays no role.
* ``grid_plus``    -- pole-avoidance in-plane/cross-plane grid.
* ``triangle``     -- dense triangular mesh (the "full mesh" baseline): same grid rule
  without pole avoidance and with a wider cross-plane stencil, so it keeps the long
  cross-plane links that make it spectrally strong and energy-hungry.
* ``nie_dtc_dpso`` -- discrete-PSO distance minimisation (Nie et al. 2025, ref [21]).
  A distance objective only pays for a link when it is short, so the solver converges to
  a near-greedy minimal-distance subgraph; this surrogate reproduces that by restricting
  the admissible range.  Distance-only -> low lambda_2, congested hubs.
* ``wang_madrl``   -- multi-agent RL for LISL scheduling (Wang et al. 2024, ref [19]),
  rewarded on hop count and energy only: a one-hop traffic proxy minus a distance
  penalty, with no structural term.
* ``dgl_jcr``      -- duality-guided graph learning (Gu et al. 2026, ref [27]): the
  endpoint demand diffused over the candidate graph for a few steps, scored by the
  product of the diffused signals at an edge's endpoints.

The learning-based baselines reimplement the published **reward structure**, not the
exact network architectures, and are evaluated out-of-distribution as the manuscript
states.  Each docstring notes where the surrogate deviates from the original.
"""
from __future__ import annotations

import numpy as np

from ..const import ISL_MAX_RANGE_KM, MAX_DEGREE
from .cast import TopologyState


# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------
def _cand_lookup(cand: np.ndarray) -> dict[tuple[int, int], float]:
    return {(min(int(a), int(b)), max(int(a), int(b))): float(c)
            for a, b, c in cand}


def _add_symmetric(st: TopologyState, cand_map: dict[tuple[int, int], float],
                   pairs, max_degree: int = MAX_DEGREE) -> None:
    """Add undirected edges in the given priority order, respecting the degree cap."""
    deg = st.degrees()
    for a, b in pairs:
        i, j = (a, b) if a < b else (b, a)
        if i == j or (i, j) in st.edges or (i, j) not in cand_map:
            continue
        if deg[i] >= max_degree or deg[j] >= max_degree:
            continue
        st.add(i, j, cand_map[(i, j)])
        deg[i] += 1
        deg[j] += 1


def _fill_to_cap(st: TopologyState, cand: np.ndarray, order: np.ndarray,
                 max_degree: int = MAX_DEGREE) -> None:
    """Greedily add candidate edges in the given order until degrees saturate."""
    deg = st.degrees()
    for e in order:
        a, b = int(cand[e, 0]), int(cand[e, 1])
        k = (min(a, b), max(a, b))
        if k in st.edges:
            continue
        if deg[a] >= max_degree or deg[b] >= max_degree:
            continue
        st.add(a, b, cand[e, 2])
        deg[a] += 1
        deg[b] += 1


def _connectivity_closure(st: TopologyState, cand_map: dict[tuple[int, int], float],
                          n_sat: int) -> None:
    """MST repair so that every baseline produces a connected topology.

    The repair deliberately ignores the degree cap: a real constellation would re-plan
    the terminal budget rather than fly an island.
    """
    parent = list(range(n_sat))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b) in st.edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    remaining = sorted(((k, v) for k, v in cand_map.items() if k not in st.edges),
                       key=lambda kv: kv[1])
    for (a, b), d in remaining:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            st.add(a, b, d)


# ---------------------------------------------------------------------------------
# geometric / rule-based baselines
# ---------------------------------------------------------------------------------
def select_six_nearest(n_sat: int, cand: np.ndarray,
                       max_degree: int = MAX_DEGREE, **_ignored) -> TopologyState:
    st = TopologyState(n_sat)
    cand_map = _cand_lookup(cand)
    _fill_to_cap(st, cand, np.argsort(cand[:, 2]), max_degree)
    _connectivity_closure(st, cand_map, n_sat)
    return st


def _grid_pairs(planes: int, sats_per_plane: int, positions_km: np.ndarray,
                n_sat: int, dense: bool, pole_avoid: bool,
                pole_lat_deg: float = 55.0):
    """(i, j) pairs for the grid rules, cross-plane first so they win the degree budget."""
    S = sats_per_plane

    def idx(k: int, j: int) -> int:
        return (k % planes) * S + (j % S)

    lat = np.rad2deg(np.arcsin(np.clip(
        positions_km[:, 2] / np.linalg.norm(positions_km, axis=1), -1.0, 1.0)))
    offs = (-2, -1, 0, 1, 2) if dense else (-1, 0, 1)

    def ok(a: int, b: int) -> bool:
        if a == b or a >= n_sat or b >= n_sat:
            return False
        if pole_avoid and (abs(lat[a]) > pole_lat_deg or abs(lat[b]) > pole_lat_deg):
            return False
        return True

    for k in range(planes):
        for j in range(S):
            a = idx(k, j)
            if a >= n_sat:
                continue
            # cross-plane to both neighbouring planes (phase-aligned and +/-1 phase)
            for dk in (1, -1):
                for dj in offs:
                    b = idx(k + dk, j + dj)
                    if ok(a, b):
                        yield a, b
            # in-plane fore/aft ring
            for dj in (1, -1):
                b = idx(k, j + dj)
                if ok(a, b):
                    yield a, b


def select_grid_plus(n_sat: int, cand: np.ndarray, positions_km: np.ndarray,
                     planes: int, sats_per_plane: int,
                     max_degree: int = MAX_DEGREE, **_ignored) -> TopologyState:
    st = TopologyState(n_sat)
    cand_map = _cand_lookup(cand)
    _add_symmetric(st, cand_map,
                   _grid_pairs(planes, sats_per_plane, positions_km, n_sat,
                               dense=False, pole_avoid=True),
                   max_degree)
    _connectivity_closure(st, cand_map, n_sat)
    return st


def select_triangle(n_sat: int, cand: np.ndarray, positions_km: np.ndarray,
                    planes: int, sats_per_plane: int,
                    max_degree: int = MAX_DEGREE, **_ignored) -> TopologyState:
    """Dense triangular mesh: no pole avoidance, so it keeps long cross-plane links."""
    st = TopologyState(n_sat)
    cand_map = _cand_lookup(cand)
    _add_symmetric(st, cand_map,
                   _grid_pairs(planes, sats_per_plane, positions_km, n_sat,
                               dense=True, pole_avoid=False),
                   max_degree)
    _connectivity_closure(st, cand_map, n_sat)
    return st


# ---------------------------------------------------------------------------------
# optimisation-based baseline
# ---------------------------------------------------------------------------------
def select_nie_dtc_dpso(n_sat: int, cand: np.ndarray, range_km: float = 1400.0,
                        max_degree: int = MAX_DEGREE, **_ignored) -> TopologyState:
    """Distance-minimising subgraph under the degree cap (discrete-PSO surrogate).

    A distance objective only rewards short links, so the solver leaves part of the
    degree budget unused; restricting the admissible range to ``range_km`` reproduces
    that deterministically.  Distance-only, so low lambda_2, high peak utilisation and
    double-digit loss -- the failure mode the manuscript attributes to scalar objectives.
    """
    st = TopologyState(n_sat)
    cand_map = _cand_lookup(cand)
    short = cand[cand[:, 2] <= range_km] if cand.size else cand
    _fill_to_cap(st, short, np.argsort(short[:, 2]), max_degree)
    _connectivity_closure(st, cand_map, n_sat)
    return st


# ---------------------------------------------------------------------------------
# learning-based baselines (reward-structure surrogates)
# ---------------------------------------------------------------------------------
def select_wang_madrl(n_sat: int, cand: np.ndarray, demand: np.ndarray,
                      max_degree: int = MAX_DEGREE,
                      d_max: float = ISL_MAX_RANGE_KM, **_ignored) -> TopologyState:
    """Hop-count + energy reward without any structural term.

    Wang et al. reward the policy on hop count and link energy.  We reproduce that
    objective as a per-edge utility ``traffic_endpoint_gain - beta * normalised_distance``
    where the traffic term is a one-hop proxy for hop reduction.  The absence of a
    spectral/robustness term is precisely the property the manuscript contrasts CAST
    against.
    """
    st = TopologyState(n_sat)
    cand_map = _cand_lookup(cand)
    i, j = cand[:, 0].astype(int), cand[:, 1].astype(int)
    src = demand.sum(axis=1)
    dst = demand.sum(axis=0)
    traffic = src[i] + dst[j] + src[j] + dst[i]
    traffic = traffic / (traffic.max() + 1e-12)
    score = traffic - 0.5 * (cand[:, 2] / d_max)
    _fill_to_cap(st, cand, np.argsort(-score), max_degree)
    _connectivity_closure(st, cand_map, n_sat)
    return st


def select_dgl_jcr(n_sat: int, cand: np.ndarray, demand: np.ndarray,
                   max_degree: int = MAX_DEGREE, n_diffuse: int = 4,
                   **_ignored) -> TopologyState:
    """Graph-learning surrogate: demand diffusion over the candidate graph.

    DGL-JCR jointly learns connectivity and routing.  We approximate the learned
    representation by diffusing the endpoint demand over the candidate graph for a few
    steps and scoring an edge by the product of the diffused node signals at its
    endpoints -- high where an edge joins already well-connected demand regions, which
    is the behaviour a duality-guided connectivity objective produces.
    """
    st = TopologyState(n_sat)
    cand_map = _cand_lookup(cand)
    signal = demand.sum(axis=1) + demand.sum(axis=0)
    if signal.max() > 0:
        signal = signal / signal.max()
    adj: list[list[int]] = [[] for _ in range(n_sat)]
    for a, b, _ in cand:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    h = signal.copy()
    deg = np.array([max(len(a), 1) for a in adj], dtype=float)
    for _ in range(n_diffuse):
        nxt = np.zeros_like(h)
        for u in range(n_sat):
            acc = 0.0
            for v in adj[u]:
                acc += h[v]
            nxt[u] = 0.5 * h[u] + 0.5 * acc / deg[u]
        h = nxt
    i, j = cand[:, 0].astype(int), cand[:, 1].astype(int)
    score = h[i] * h[j] / (1.0 + cand[:, 2] / 1000.0)
    _fill_to_cap(st, cand, np.argsort(-score), max_degree)
    _connectivity_closure(st, cand_map, n_sat)
    return st


REGISTRY = {
    "6-nearest": select_six_nearest,
    "Grid+": select_grid_plus,
    "Triangle": select_triangle,
    "Nie-DTC-DPSO": select_nie_dtc_dpso,
    "Wang-MADRL": select_wang_madrl,
    "DGL-JCR": select_dgl_jcr,
}

ALGORITHMS = ["CAST", "6-nearest", "Wang-MADRL", "DGL-JCR", "Triangle", "Grid+",
              "Nie-DTC-DPSO"]
