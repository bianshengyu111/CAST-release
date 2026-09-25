# -*- coding: utf-8 -*-
"""Walker-delta constellation generation and analytic propagation.

The manuscript uses a Starlink Shell-3-like Walker-delta constellation
(72 planes x 20 satellites, 1,401 active, h = 550 km, i = 53 deg).  Positions in
the paper come from SGP4 propagation of public TLEs; for a self-contained release we
provide two options:

* ``propagator="kepler"`` (default) -- a closed-form circular Walker-delta propagator.
  Deterministic, dependency-free, and sufficient for topology experiments because the
  topology decision depends on *relative* inter-satellite geometry, not on absolute
  ephemeris accuracy.
* ``propagator="sgp4"`` -- real SGP4 propagation from a user-supplied TLE file
  (``sgp4`` must be installed separately).  Selected automatically when a TLE file
  is supplied via ``tle_path``.

Both paths expose the same interface, so the rest of the pipeline is agnostic to the
choice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..const import MU_EARTH, OMEGA_EARTH, R_EARTH_KM


@dataclass
class WalkerDelta:
    """A Walker-delta constellation with an explicit active-satellite count."""

    planes: int = 72
    sats_per_plane: int = 20
    active: int = 1401
    altitude_km: float = 550.0
    inclination_deg: float = 53.0
    phasing: int = 1
    propagator: str = "kepler"
    tle_path: Optional[str] = None

    raan0: np.ndarray = field(init=False, repr=False)
    m0: np.ndarray = field(init=False, repr=False)
    period_s: float = field(init=False)

    def __post_init__(self) -> None:
        total = self.planes * self.sats_per_plane
        if self.active > total:
            raise ValueError(f"active={self.active} exceeds planes*sats_per_plane={total}")
        # Walker-delta phasing: RAAN_k = 2*pi*k/P, M_jk = 2*pi*j/S + F*2*pi*k/T.
        k = np.arange(self.planes)
        j = np.arange(self.sats_per_plane)
        raan = 2.0 * np.pi * k / self.planes
        m = (2.0 * np.pi * j[None, :] / self.sats_per_plane
             + self.phasing * 2.0 * np.pi * k[:, None] / total)
        self.raan_all = np.repeat(raan, self.sats_per_plane)
        self.m_all = m.reshape(-1)
        # Keep only the active subset (default: first N slots of the pattern).
        self.raan0 = self.raan_all[: self.active].copy()
        self.m0 = self.m_all[: self.active].copy()
        a = R_EARTH_KM + self.altitude_km
        self.sma_km = a
        self.mean_motion = np.sqrt(MU_EARTH / a ** 3)          # rad / s
        self.period_s = 2.0 * np.pi / self.mean_motion
        self.inclination = np.deg2rad(self.inclination_deg)
        self._tle_sat = None
        if self.propagator == "sgp4" and self.tle_path:
            self._init_sgp4(self.tle_path)

    # -- propagators ---------------------------------------------------------------
    def positions_eci(self, t_s: float) -> np.ndarray:
        """Return ``(N, 3)`` ECI positions [km] at simulation time ``t_s``."""
        if self.propagator == "sgp4" and self._tle_sat is not None:
            return self._positions_sgp4(t_s)
        return self._positions_kepler(t_s)

    def _positions_kepler(self, t_s: float) -> np.ndarray:
        u = self.m0 + self.mean_motion * t_s          # argument of latitude (circular)
        ci, si = np.cos(self.inclination), np.sin(self.inclination)
        cu, su = np.cos(u), np.sin(u)
        x = self.sma_km * cu
        y = self.sma_km * su * ci
        z = self.sma_km * su * si
        # rotate by RAAN about the z axis
        cr, sr = np.cos(self.raan0), np.sin(self.raan0)
        xr = cr * x - sr * y
        yr = sr * x + cr * y
        return np.column_stack([xr, yr, z])

    def _init_sgp4(self, tle_path: str) -> None:
        from sgp4.api import Satrec, jday  # imported lazily so the dep stays optional
        recs = []
        with open(tle_path, "r", encoding="utf-8") as fh:
            lines = [ln.rstrip() for ln in fh if ln.strip()]
        for i in range(0, len(lines) - 2, 3):
            recs.append(Satrec.twoline2rv(lines[i + 1], lines[i + 2]))
        self._tle_sat = recs[: self.active]
        if not self._tle_sat:
            raise ValueError(f"no TLE records parsed from {tle_path}")

    def _positions_sgp4(self, t_s: float) -> np.ndarray:
        from sgp4.api import jday
        # fixed epoch for reproducibility: 2026-01-01 00:00:00 UTC
        jd, fr = jday(2026, 1, 1, 0, 0, t_s % 86400.0)
        out = np.empty((len(self._tle_sat), 3))
        for i, sat in enumerate(self._tle_sat):
            e, r, _ = sat.sgp4(jd, fr)
            out[i] = r if e == 0 else np.nan
        return out

    # -- frames --------------------------------------------------------------------
    def positions_ecef(self, t_s: float) -> np.ndarray:
        """ECI positions rotated into the Earth-fixed frame (for city geometry)."""
        eci = self.positions_eci(t_s)
        theta = OMEGA_EARTH * t_s
        ct, st_ = np.cos(theta), np.sin(theta)
        x = ct * eci[:, 0] + st_ * eci[:, 1]
        y = -st_ * eci[:, 0] + ct * eci[:, 1]
        return np.column_stack([x, y, eci[:, 2]])

    def orbit_period_min(self) -> float:
        return self.period_s / 60.0


def walker_72x20(**kwargs) -> WalkerDelta:
    """Convenience constructor for the manuscript's primary constellation.

    Any keyword (``planes``, ``sats_per_plane``, ``active``, ``propagator``, ...)
    overrides the default, so reduced-size smoke configurations are one call away.
    """
    defaults = dict(planes=72, sats_per_plane=20, active=1401,
                    altitude_km=550.0, inclination_deg=53.0)
    defaults.update(kwargs)
    return WalkerDelta(**defaults)


def walker_kuiper_like(**kwargs) -> WalkerDelta:
    """Kuiper-like inclined configuration used in the generality study (Sec. IV-J)."""
    defaults = dict(planes=34, sats_per_plane=34, active=1156,
                    altitude_km=630.0, inclination_deg=51.9)
    defaults.update(kwargs)
    return WalkerDelta(**defaults)
