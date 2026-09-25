# -*- coding: utf-8 -*-
"""Candidate inter-satellite-link generation.

A pair (i, j) is an admissible candidate in slot t when

* the line-of-sight chord is at most ``d_max`` (manuscript Eq. 1), and
* the chord does not intersect the Earth (the horizon-limited chord at h = 550 km
  is ~5,408 km, so d_max = 5,000 km already implies visibility with margin).

The candidate set is recomputed every slot from the propagated positions and is
shared, unchanged, by CAST and every baseline (manuscript Sec. IV-A, "our code makes
the decisions").
"""
from __future__ import annotations

import numpy as np

from ..const import ISL_MAX_RANGE_KM, R_EARTH_KM


def chord_distances(positions_km: np.ndarray) -> np.ndarray:
    """Full pairwise chord-distance matrix [km]."""
    p = np.asarray(positions_km, dtype=float)
    diff = p[:, None, :] - p[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def candidate_edges(positions_km: np.ndarray, d_max: float = ISL_MAX_RANGE_KM,
                    earth_radius_km: float = R_EARTH_KM) -> np.ndarray:
    """Return an ``(M, 3)`` array of ``[i, j, d_ij]`` candidate edges (i < j)."""
    n = positions_km.shape[0]
    dist = chord_distances(positions_km)
    iu, ju = np.triu_indices(n, k=1)
    d = dist[iu, ju]
    keep = (d > 0.0) & (d <= d_max)

    # Earth-occlusion check: the closest approach of the chord to the Earth centre
    # must exceed R_E.  Equivalent to testing the triangle altitude.
    pi, pj = positions_km[iu[keep]], positions_km[ju[keep]]
    dij = d[keep]
    # projection length from pi onto the (pi->pj) direction
    t = -np.einsum("ij,ij->i", pi, pj - pi) / (dij ** 2)
    t = np.clip(t, 0.0, 1.0)
    closest = pi + t[:, None] * (pj - pi)
    occluded = np.linalg.norm(closest, axis=1) < earth_radius_km
    keep_idx = np.flatnonzero(keep)[~occluded]

    edges = np.column_stack([iu[keep_idx], ju[keep_idx], dist[iu[keep_idx], ju[keep_idx]]])
    return edges


def feasible_mask_for_pair(positions_km: np.ndarray, i: int, j: int,
                           d_max: float = ISL_MAX_RANGE_KM) -> bool:
    d = float(np.linalg.norm(positions_km[i] - positions_km[j]))
    if d == 0.0 or d > d_max:
        return False
    # chord-to-centre distance
    t = -float(np.dot(positions_km[i], positions_km[j] - positions_km[i])) / (d ** 2)
    t = min(1.0, max(0.0, t))
    closest = positions_km[i] + t * (positions_km[j] - positions_km[i])
    return float(np.linalg.norm(closest)) >= R_EARTH_KM
