# -*- coding: utf-8 -*-
"""End-to-end simulation driver: one (algorithm, flow count, seed) configuration.

Pipeline per 30-s slot (manuscript Sec. IV-A):

    positions -> candidate edges -> access-star mapping -> demand matrix D_sr(t)
    -> topology selection -> congestion-aware routing -> queueing evaluation -> KPIs

The topology state is carried across slots so that CAST's warm start / passive removal
and the baselines' per-slot rules see the previous slot, exactly as in Algorithm 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..const import MAX_DEGREE, SEC_PER_SLOT
from ..metrics.kpi import ConfigKPIs, observations_to_kpis
from ..routing.congestion_aware import route_commodities
from ..spectral.fiedler import algebraic_connectivity
from ..simulate.discrete_event import evaluate_slot
from ..topology import baselines
from ..topology.access_mapping import assign_access_stars
from ..topology.cast import TopologyState, select_cast
from ..topology.visibility import candidate_edges
from ..traffic.gravity import (CITY_LATLON, CITY_POP, demand_matrix,
                               sample_flows)

# Learning-based baselines evaluated out-of-distribution (manuscript Sec. IV-A).
OOD_BASELINES = {"Wang-MADRL", "DGL-JCR"}


@dataclass
class RunConfig:
    algorithm: str = "CAST"
    n_flows: int = 5000
    seed: int = 1
    n_slots: int = 60
    ca2: float = 4.0
    alpha_d: float = 0.7
    alpha_c: float = 0.3
    eta: tuple[float, ...] = (0.30, 0.25, 0.25, 0.20)
    delta: float = 0.05
    max_iter: int = 10
    kappa: float = 0.05
    regime: str = "dynamic"       # "static" (fixed access mapping) or "dynamic"
    mst_init: bool = True         # False -> random-spanning cold start (ablation)
    routing_batch: int = 64       # commodities sharing one routing weight snapshot


def _select_topology(cfg: RunConfig, walker, cand, positions_eci, demand, prev_state,
                     planes, sats_per_plane):
    if cfg.algorithm == "CAST":
        st, diag = select_cast(
            walker.active, positions_eci, demand, prev_state=prev_state,
            alpha_d=cfg.alpha_d, alpha_c=cfg.alpha_c, eta=cfg.eta,
            delta=cfg.delta, max_iter=cfg.max_iter, mst_init=cfg.mst_init,
            routing_batch=cfg.routing_batch)
        return st, diag
    fn = baselines.REGISTRY.get(cfg.algorithm)
    if fn is None:
        raise ValueError(f"unknown algorithm {cfg.algorithm!r}")
    kwargs = dict(n_sat=walker.active, cand=cand, max_degree=MAX_DEGREE)
    # pass what each rule needs
    import inspect
    params = inspect.signature(fn).parameters
    if "positions_km" in params:
        kwargs["positions_km"] = positions_eci
    if "planes" in params:
        kwargs["planes"] = planes
        kwargs["sats_per_plane"] = sats_per_plane
    if "demand" in params:
        kwargs["demand"] = demand
    return fn(**kwargs), {}


def run_configuration(cfg: RunConfig, walker, planes: int, sats_per_plane: int,
                      progress=None) -> ConfigKPIs:
    """Run one full 60-slot scenario and return aggregated KPIs."""
    flows = sample_flows(cfg.n_flows, cfg.seed)
    # Out-of-distribution demand for the learning-based baselines: the manuscript's RL
    # policies were trained on traffic distributions 1-6 and tested on 7-10, so their
    # demand signal is drawn from a different distribution than the one being evaluated.
    # Geometry-only and rule-based baselines are unaffected (they never see demand), and
    # CAST sees the true per-slot D(t) directly.
    # a different demand *geography* (populations rolled by 7 cities): a policy
    # trained on the original distribution sees a misaligned hotspot map
    flows_ood = sample_flows(cfg.n_flows, cfg.seed + 10_000,
                             pop=np.roll(CITY_POP, 7))
    prev_state = None
    fixed_access = None
    obs_list = []

    for slot in range(cfg.n_slots):
        t_s = slot * SEC_PER_SLOT
        pos_eci = walker.positions_eci(t_s)
        pos_ecef = walker.positions_ecef(t_s)
        src_sat = assign_access_stars(CITY_LATLON, pos_ecef)
        if cfg.regime == "static" and fixed_access is not None:
            src_sat = fixed_access
        elif fixed_access is None and cfg.regime == "static":
            fixed_access = src_sat
        dst_sat = src_sat
        demand = demand_matrix(flows, src_sat, dst_sat, walker.active)

        cand = candidate_edges(pos_eci)
        policy_demand = (demand_matrix(flows_ood, src_sat, dst_sat, walker.active)
                         if cfg.algorithm in OOD_BASELINES else demand)
        st, _diag = _select_topology(cfg, walker, cand, pos_eci, policy_demand,
                                     prev_state, planes, sats_per_plane)
        prev_state = st if cfg.algorithm == "CAST" else prev_state

        idx, dist = st.index_and_dist()
        commodities = [(int(s), int(d), float(demand[s, d]))
                       for s, d in np.argwhere(demand > 0)]
        res = route_commodities(walker.active, idx, dist, commodities,
                                alpha_d=cfg.alpha_d, alpha_c=cfg.alpha_c,
                                batch_size=cfg.routing_batch)

        edge_lookup = {(min(int(a), int(b)), max(int(a), int(b))): float(c)
                       for a, b, c in zip(idx[:, 0], idx[:, 1], dist)}
        obs = evaluate_slot(res.paths, res.commodities, edge_lookup, res.utilization,
                            walker.active, ca2=cfg.ca2, kappa=cfg.kappa)
        obs.lambda2 = algebraic_connectivity(
            walker.active, [(int(a), int(b)) for a, b in idx])
        # unreachable commodities reduce reachability
        if res.unrouted:
            obs.reachability = max(0.0, 1.0 - res.unrouted / max(len(commodities), 1))
        obs_list.append(obs)
        if progress:
            progress(slot, obs)

    return observations_to_kpis(obs_list)
