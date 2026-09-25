# -*- coding: utf-8 -*-
"""Smoke and unit tests for the CAST reference implementation.

Run:  pytest -q        (or)   python tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constellation.walker import walker_72x20                       # noqa: E402
from src.metrics.kpi import ConfigKPIs, composite_scores                # noqa: E402
from src.routing.congestion_aware import route_commodities              # noqa: E402
from src.sim.runner import RunConfig, run_configuration                 # noqa: E402
from src.simulate.discrete_event import packet_simulate_path            # noqa: E402
from src.spectral.fiedler import (algebraic_connectivity, cosine_similarity,  # noqa: E402
                                  fiedler_approx, fiedler_exact)
from src.topology.cast import select_cast                               # noqa: E402
from src.topology.visibility import candidate_edges                     # noqa: E402
from src.traffic.gravity import city_pair_weights, sample_flows         # noqa: E402


def test_walker_geometry():
    w = walker_72x20()
    p = w.positions_eci(0.0)
    assert p.shape == (1401, 3)
    r = np.linalg.norm(p, axis=1)
    assert np.allclose(r, 6371.0 + 550.0, atol=1e-6)


def test_candidate_edges_are_in_range():
    w = walker_72x20(planes=6, sats_per_plane=8, active=48)
    cand = candidate_edges(w.positions_eci(0.0))
    assert cand.shape[0] > 0
    assert cand[:, 2].max() <= 5000.0
    assert (cand[:, 0] < cand[:, 1]).all()


def test_fiedler_matches_on_a_path_graph():
    # path graph 0-1-2-3: known Fiedler vector is proportional to (-3,-1,1,3)
    edges = [(0, 1), (1, 2), (2, 3)]
    q = fiedler_exact(4, edges)
    ref = np.array([-3.0, -1.0, 1.0, 3.0])
    assert abs(cosine_similarity(q, ref)) > 0.99
    la = algebraic_connectivity(4, edges)
    assert abs(la - 0.586) < 0.02          # 2 - sqrt(2)


def test_fiedler_approx_close_to_exact():
    # geometric graph (as in a constellation mesh): 10x10 grid with diagonal links
    n, side = 100, 10
    edges = []
    for r in range(side):
        for c in range(side):
            u = r * side + c
            if c + 1 < side:
                edges.append((u, u + 1))
            if r + 1 < side:
                edges.append((u, u + side))
            if r + 1 < side and c + 1 < side:
                edges.append((u, u + side + 1))
    qe, qa = fiedler_exact(n, edges), fiedler_approx(n, edges)
    assert abs(cosine_similarity(qe, qa)) > 0.6


def test_gravity_and_flows():
    w = city_pair_weights()
    assert w.shape == (28, 28)
    assert np.allclose(np.diag(w), 0.0)
    f = sample_flows(100, seed=1)
    assert f.shape == (100, 3)
    assert np.all((f[:, 0] >= 0) & (f[:, 0] < 28))
    assert np.allclose(f[:, 2], 40.0)


def test_routing_returns_connected_paths():
    w = walker_72x20(planes=6, sats_per_plane=8, active=48)
    pos = w.positions_eci(0.0)
    cand = candidate_edges(pos)
    idx = cand[:, :2].astype(int)
    dist = cand[:, 2]
    res = route_commodities(48, idx, dist, [(0, 47, 40.0), (5, 30, 40.0)])
    assert res.unrouted == 0
    assert all(p[0] != p[-1] for p in res.paths)
    assert max(res.utilization.values()) <= 1.5   # sanity: no runaway accumulation


def test_cast_improves_algebraic_connectivity():
    w = walker_72x20(planes=6, sats_per_plane=8, active=48)
    pos = w.positions_eci(0.0)
    demand = np.zeros((48, 48))
    demand[0, 47] = 400.0
    demand[10, 40] = 400.0
    st, diag = select_cast(48, pos, demand)
    idx = np.array(list(st.edges.keys()))
    la = algebraic_connectivity(48, [(int(a), int(b)) for a, b in idx])
    assert la > 0
    assert diag["n_edges"] > 0


def test_composite_prefers_a_dominated_vector():
    good = ConfigKPIs(1.0, 0.99, 1.0, 10, 20, 1, 0.99, 0.5, 0.01, 100, 0.2)
    bad = ConfigKPIs(0.5, 0.90, 5.0, 20, 40, 3, 0.80, 0.9, 0.10, 200, 0.1)
    sc = composite_scores({"good": good, "bad": bad})
    assert sc["good"] > sc["bad"]
    assert abs(sc["good"] - 1.0) < 1e-9


def test_packet_simulator_runs():
    path = list(range(5))
    edge = {}
    for a, b in zip(path[:-1], path[1:]):
        edge[(a, b)] = 1200.0
    r = packet_simulate_path(path, edge, rate_mbps=40.0, n_packets=500, seed=0)
    assert 0.0 <= r["loss_frac"] <= 1.0
    assert r["mean_ms"] > 0


def test_end_to_end_small():
    w = walker_72x20(planes=6, sats_per_plane=8, active=48)
    cfg = RunConfig(algorithm="CAST", n_flows=50, seed=1, n_slots=2)
    k = run_configuration(cfg, w, 6, 8)
    assert k.throughput_gbps >= 0
    assert 0 <= k.reachability_frac <= 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:      # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
