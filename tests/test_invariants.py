# -*- coding: utf-8 -*-
"""Measurement-pipeline invariant tests (second-round review checklist).

These assertions encode constraints that a correct pipeline run can never violate.
Every failure mode flagged in the second-round review is covered by one of them:

1. queueing delay   <=  buffer x service time x hops          (per flow, every run)
2. peak utilisation >=  mean utilisation of any link subset   (top-5% share identity)
3. delivered + lost == offered                                (flow conservation)
4. the routing weight Psi(rho) = rho / (1 - rho + eps) is finite wherever it is used
   and reported utilisation statistics never exceed 1
plus the delay-decomposition identity and topology-stability checks (the static
intra-plane ring survives passive removal, algebraic connectivity stays positive).

Run standalone (``python tests/test_invariants.py``) or under pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.const import (BUFFER_PACKETS, SEC_PER_SLOT, SERVICE_TIME_S)  # noqa: E402
from src.constellation.walker import (static_intra_plane_edges,       # noqa: E402
                                      walker_72x20)
from src.routing.congestion_aware import _psi, route_commodities     # noqa: E402
from src.spectral.fiedler import algebraic_connectivity               # noqa: E402
from src.simulate.discrete_event import evaluate_slot                 # noqa: E402
from src.topology.access_mapping import assign_access_stars           # noqa: E402
from src.topology.cast import select_cast                             # noqa: E402
from src.traffic.gravity import CITY_LATLON, demand_matrix, sample_flows  # noqa: E402

# reduced but real configuration: 8 planes x 12 satellites, 300 flows, 4 slots
PLANES, SPP, NF, NSLOTS, SEED = 8, 12, 300, 4, 1


def _mini_run(planes: int = PLANES, sats_per_plane: int = SPP, active: int | None = None,
              n_slots: int = NSLOTS, n_flows: int = NF, seed: int = SEED):
    if active is None:
        active = planes * sats_per_plane
    walker = walker_72x20(planes=planes, sats_per_plane=sats_per_plane, active=active)
    static_edges = static_intra_plane_edges(planes, sats_per_plane, active)
    flows = sample_flows(n_flows, seed)
    prev = None
    obs_list, edge_counts, lam2s = [], [], []
    for slot in range(NSLOTS):
        pos_eci = walker.positions_eci(slot * SEC_PER_SLOT)
        pos_ecef = walker.positions_ecef(slot * SEC_PER_SLOT)
        src = assign_access_stars(CITY_LATLON, pos_ecef)
        demand = demand_matrix(flows, src, src, walker.active)
        st, _ = select_cast(walker.active, pos_eci, demand, prev_state=prev,
                            static_edges=static_edges)
        idx, dist = st.index_and_dist()
        comms = [(int(s), int(d), float(demand[s, d]))
                 for s, d in np.argwhere(demand > 0)]
        res = route_commodities(walker.active, idx, dist, comms)
        edge_lookup = {(min(int(a), int(b)), max(int(a), int(b))): float(c)
                       for a, b, c in zip(idx[:, 0], idx[:, 1], dist)}
        obs = evaluate_slot(res.paths, res.commodities, edge_lookup, res.utilization,
                            walker.active)
        obs.lambda2 = algebraic_connectivity(
            walker.active, [(int(a), int(b)) for a, b in idx])
        obs_list.append(obs)
        edge_counts.append(len(st.edges))
        lam2s.append(obs.lambda2)
        prev = st
    return obs_list, edge_counts, lam2s, len(static_edges)


_OBS, _EDGES, _LAM2, _N_STATIC = _mini_run()


def test_queueing_delay_bounded():
    """R1 check 1: no packet waits longer than buffer x service time per hop."""
    cap_per_hop_ms = BUFFER_PACKETS * SERVICE_TIME_S * 1e3      # 58.6 us in ms
    worst = 0.0
    for obs in _OBS:
        for f in obs.flows:
            worst = max(worst, f.t_queue_ms)
            assert f.t_queue_ms <= f.hops * cap_per_hop_ms + 1e-9, (
                f"T_queue {f.t_queue_ms:.6f} ms exceeds {f.hops} hops x "
                f"{cap_per_hop_ms:.6f} ms")
    assert worst > 0.0  # the bound is exercised, not vacuous


def test_delay_decomposition_identity():
    """mean_delay == T_prop + T_trans + T_queue for every flow record."""
    for obs in _OBS:
        for f in obs.flows:
            lhs = f.t_prop_ms + f.t_trans_ms + f.t_queue_ms
            assert abs(lhs - f.mean_delay_ms) <= 1e-6 * max(1.0, f.mean_delay_ms)
            assert f.tail_delay_ms >= f.mean_delay_ms - 1e-9
            assert 0.0 <= f.loss_frac <= 1.0
            assert f.jitter_ms >= 0.0


def test_flow_conservation():
    """R1 check 3: delivered + lost == offered, per slot and in aggregate."""
    for obs in _OBS:
        offered = sum(f.offered_mbps for f in obs.flows)
        delivered = sum(f.delivered_mbps for f in obs.flows)
        lost = sum(f.offered_mbps * f.loss_frac for f in obs.flows)
        assert abs((delivered + lost) - offered) <= 1e-6 * offered
        assert obs.loss_rate <= 1.0


def test_top5_share_vs_peak():
    """R1 check 2: mean utilisation of the busiest 5% of links can never exceed the
    peak utilisation: share / 0.05 x mean_util <= peak_util."""
    for n, obs in enumerate(_OBS):
        if obs.mean_util <= 0.0:
            continue
        implied = obs.top5pct_traffic_share / 0.05 * obs.mean_util
        assert implied <= obs.peak_util + 1e-9, (
            f"slot {n}: top-5% share {obs.top5pct_traffic_share:.4f} implies mean "
            f"utilisation {implied:.4f} > peak {obs.peak_util:.4f}")
        # the direct identity on the same utilisation vector
        assert obs.peak_util >= obs.mean_util - 1e-12


def test_utilisation_statistics_bounded():
    """R1 check 4: reported utilisation never exceeds 1; Psi stays finite; links
    whose offered load exceeds capacity are reported, not hidden."""
    for obs in _OBS:
        assert 0.0 <= obs.peak_util <= 1.0
        assert 0.0 <= obs.mean_util <= 1.0
        assert 0.0 <= obs.oversub_link_frac <= 1.0
    for rho in (0.0, 0.5, 0.9, 0.99, 1.0 - 1e-12):
        assert np.isfinite(_psi(rho))


def test_topology_does_not_collapse():
    """The static intra-plane ring survives passive removal and the graph stays
    connected, so long runs cannot shed their backbone."""
    for n, (m, lam) in enumerate(zip(_EDGES, _LAM2)):
        assert m >= _N_STATIC, (
            f"slot {n}: {m} active edges < {_N_STATIC} static ring edges")
        assert lam > 0.0, f"slot {n}: graph disconnected (lambda2 = {lam})"


def test_topology_connected_with_partial_plane():
    """A satellite whose whole plane is inactive (no intra-plane ring at all) can only
    stay connected through dynamic inter-plane ISLs -- passive removal must never cut
    it off.  active=85 leaves the last plane of the 8x12 grid with a single satellite."""
    _, edges, lam2s, _ = _mini_run(planes=8, sats_per_plane=12, active=85,
                                   n_slots=3, n_flows=150, seed=2)
    for n, lam in enumerate(lam2s):
        assert lam > 0.0, f"slot {n}: partial-plane satellite disconnected (lambda2=0)"


def main() -> int:
    tests = [test_queueing_delay_bounded,
             test_delay_decomposition_identity,
             test_flow_conservation,
             test_top5_share_vs_peak,
             test_utilisation_statistics_bounded,
             test_topology_does_not_collapse,
             test_topology_connected_with_partial_plane]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
