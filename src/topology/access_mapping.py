# -*- coding: utf-8 -*-
"""City -> access-satellite mapping.

Rule (manuscript Sec. III-D and author note 08_city_access_mapping.md): for each
city and slot, the access satellite is the visible satellite whose elevation above
the local horizon is highest, provided that elevation is at least 10 deg.  Ties
(within 0.1 deg) are broken by the smaller azimuth difference from due south.  The
mapping is recomputed every 30-s slot; ~10-15 of the 28 cities switch access star in
a typical slot, which is the sole source of time-variation in D_sr(t).
"""
from __future__ import annotations

import numpy as np

from ..const import MIN_ELEVATION_DEG, R_EARTH_KM


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> np.ndarray:
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    r = R_EARTH_KM + alt_km
    return np.array([r * np.cos(lat) * np.cos(lon),
                     r * np.cos(lat) * np.sin(lon),
                     r * np.sin(lat)])


def elevation_deg(sat_ecef_km: np.ndarray, site_ecef_km: np.ndarray) -> np.ndarray:
    """Elevation [deg] of each satellite as seen from a ground site."""
    los = sat_ecef_km - site_ecef_km[None, :]
    rng = np.linalg.norm(los, axis=1)
    up = site_ecef_km / np.linalg.norm(site_ecef_km)
    sin_el = (los @ up) / np.maximum(rng, 1e-9)
    return np.rad2deg(np.arcsin(np.clip(sin_el, -1.0, 1.0)))


def azimuth_deg(sat_ecef_km: np.ndarray, site_ecef_km: np.ndarray) -> np.ndarray:
    """Azimuth [deg, 0 = north, clockwise] of each satellite from a ground site."""
    lat = np.arcsin(site_ecef_km[2] / np.linalg.norm(site_ecef_km))
    lon = np.arctan2(site_ecef_km[1], site_ecef_km[0])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    los = sat_ecef_km - site_ecef_km[None, :]
    return np.rad2deg(np.arctan2(los @ east, los @ north)) % 360.0


def assign_access_stars(cities_latlon: np.ndarray, sat_ecef_km: np.ndarray,
                        min_elev_deg: float = MIN_ELEVATION_DEG) -> np.ndarray:
    """Return, for each city, the index of its access satellite (-1 if none visible)."""
    n_sat = sat_ecef_km.shape[0]
    out = np.full(cities_latlon.shape[0], -1, dtype=int)
    for c, (lat, lon) in enumerate(cities_latlon):
        site = geodetic_to_ecef(float(lat), float(lon))
        el = elevation_deg(sat_ecef_km, site)
        visible = el >= min_elev_deg
        if not np.any(visible):
            continue
        best = np.flatnonzero(visible)
        # primary key: highest elevation; tie-break: azimuth closest to due south (180)
        az = azimuth_deg(sat_ecef_km[best], site)
        order = np.lexsort((np.abs(az - 180.0), -el[best]))
        out[c] = best[order[0]]
    return out
