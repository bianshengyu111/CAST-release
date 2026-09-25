# -*- coding: utf-8 -*-
"""Traffic demand: 28-city gravity model and per-flow generation.

Manuscript Sec. IV-A: traffic originates from 28 global cities under a gravity model
``P_i P_j / d_ij^geo`` (UN World Urbanization Prospects 2024), swept from 500 to
5,000 flows at 40 Mbps per flow.  Per-city offered load is constant across the 60
slots; the time-variation of ``D_sr(t)`` comes entirely from the access-star
handovers (see ``access_mapping``).

Flow generation is deterministic given ``(n_flows, seed)`` so that every algorithm
sees exactly the same flow set in a given configuration -- a prerequisite for a fair
comparison.
"""
from __future__ import annotations

import numpy as np

from ..const import FLOW_RATE_MBPS, R_EARTH_KM

# 28 cities: (name, lat_deg, lon_deg, population_millions).
# Populations are order-of-magnitude UN WUP style figures; they set only the
# *relative* weights of the gravity model, which is what the topology reacts to.
CITIES = [
    ("New York",     40.71,  -74.01, 18.8),
    ("Los Angeles",  34.05, -118.24, 12.5),
    ("Chicago",      41.88,  -87.63,  8.9),
    ("Toronto",      43.65,  -79.38,  6.2),
    ("Mexico City",  19.43,  -99.13, 21.8),
    ("Sao Paulo",   -23.55,  -46.63, 22.4),
    ("Buenos Aires",-34.60,  -58.38, 15.4),
    ("Lima",        -12.05,  -77.04, 11.0),
    ("London",       51.51,   -0.13, 14.3),
    ("Paris",        48.86,    2.35, 11.1),
    ("Madrid",       40.42,   -3.70,  6.7),
    ("Moscow",       55.75,   37.62, 12.6),
    ("Istanbul",     41.01,   28.98, 15.8),
    ("Cairo",        30.04,   31.24, 21.3),
    ("Lagos",         6.52,    3.38, 14.9),
    ("Johannesburg",-26.20,   28.05,  5.9),
    ("Nairobi",      -1.29,   36.82,  5.1),
    ("Dubai",        25.20,   55.27,  3.5),
    ("Mumbai",       19.08,   72.88, 20.4),
    ("Delhi",        28.61,   77.21, 29.6),
    ("Bangkok",      13.76,  100.50, 10.6),
    ("Singapore",     1.35,  103.82,  5.9),
    ("Jakarta",      -6.21,  106.85, 10.6),
    ("Hong Kong, China", 22.32, 114.17,  7.5),
    ("Shanghai",     31.23,  121.47, 27.1),
    ("Beijing",      39.90,  116.41, 21.5),
    ("Tokyo",        35.68,  139.69, 37.4),
    ("Sydney",      -33.87,  151.21,  5.3),
]

CITY_NAMES = [c[0] for c in CITIES]
CITY_LATLON = np.array([[c[1], c[2]] for c in CITIES], dtype=float)
CITY_POP = np.array([c[3] for c in CITIES], dtype=float)


def great_circle_km(latlon_a: np.ndarray, latlon_b: np.ndarray) -> np.ndarray:
    """Great-circle distance matrix [km] between two sets of (lat, lon)."""
    la, lo = np.deg2rad(latlon_a[:, 0]), np.deg2rad(latlon_a[:, 1])
    lb, lob = np.deg2rad(latlon_b[:, 0]), np.deg2rad(latlon_b[:, 1])
    cos = (np.sin(la)[:, None] * np.sin(lb)[None, :]
           + np.cos(la)[:, None] * np.cos(lb)[None, :] * np.cos(lo[:, None] - lob[None, :]))
    return R_EARTH_KM * np.arccos(np.clip(cos, -1.0, 1.0))


def city_pair_weights(pop: np.ndarray = CITY_POP,
                      latlon: np.ndarray = CITY_LATLON) -> np.ndarray:
    """Gravity weight ``P_i P_j / d_ij`` for every ordered city pair (zero diagonal)."""
    d = great_circle_km(latlon, latlon)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = pop[:, None] * pop[None, :] / np.where(d > 1.0, d, np.inf)
    w[~np.isfinite(w)] = 0.0
    np.fill_diagonal(w, 0.0)
    return w


def sample_flows(n_flows: int, seed: int, rate_mbps: float = FLOW_RATE_MBPS,
                 pop: np.ndarray | None = None):
    """Sample ``(src_city, dst_city, rate_mbps)`` triples from the gravity model.

    Returns an ``(n_flows, 3)`` float array whose first two columns are integer city
    indices stored as floats for vectorised downstream use.  ``pop`` overrides the city
    population vector, which is how the out-of-distribution traffic patterns used for
    the learning-based baselines are generated.
    """
    rng = np.random.default_rng(seed)
    w = city_pair_weights(CITY_POP if pop is None else pop)
    p = (w / w.sum()).ravel()
    picked = rng.choice(p.size, size=n_flows, replace=True, p=p)
    src, dst = np.divmod(picked, w.shape[1])
    rates = np.full(n_flows, float(rate_mbps))
    return np.column_stack([src.astype(float), dst.astype(float), rates])


def demand_matrix(flows: np.ndarray, src_sat: np.ndarray, dst_sat: np.ndarray,
                  n_sat: int) -> np.ndarray:
    """Aggregate flows into ``D_sr(t)`` keyed by (source satellite, dest satellite).

    ``src_sat`` / ``dst_sat`` map each city to its current access satellite; a flow is
    dropped when either endpoint has no visible access star (index -1).
    """
    d = np.zeros((n_sat, n_sat))
    src = src_sat[flows[:, 0].astype(int)]
    dst = dst_sat[flows[:, 1].astype(int)]
    ok = (src >= 0) & (dst >= 0) & (src != dst)
    np.add.at(d, (src[ok], dst[ok]), flows[ok, 2])
    return d
