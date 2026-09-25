# -*- coding: utf-8 -*-
"""Physical and system constants used throughout the CAST reference implementation.

Every value here is traceable to the manuscript (IJSCN 5182484, revised v26).
Keeping them in one module makes the simulation auditable: a reader can check any
reported number against a named constant instead of a magic literal buried in the
algorithm code.
"""

# --- Earth & orbit -----------------------------------------------------------------
R_EARTH_KM = 6371.0          # mean Earth radius, manuscript Sec. III-A
MU_EARTH = 398600.4418       # km^3 / s^2, WGS-84 gravitational parameter
OMEGA_EARTH = 7.2921159e-5   # rad / s, Earth rotation rate (ECI -> ECEF)
C_KM_S = 299792.458          # km / s, speed of light in vacuum
SEC_PER_SLOT = 30.0          # manuscript Eq. (setup): slot duration tau

# --- Link layer --------------------------------------------------------------------
ISL_CAPACITY_MBPS = 20480.0  # C_e, all ISLs
ISL_MAX_RANGE_KM = 5000.0    # d_max, Eq. (1)
MAX_DEGREE = 6               # Delta_max, optical terminals per platform
PACKET_BITS = 12000.0        # 1,500 B fixed packets, Sec. III-E
BUFFER_PACKETS = 100         # FIFO output queue depth per ISL direction, Sec. III-E
SERVICE_TIME_S = PACKET_BITS / (ISL_CAPACITY_MBPS * 1e6)  # 0.586 us

# --- Energy model ------------------------------------------------------------------
KAPPA_W_PER_KM = 0.05        # equivalent marginal coefficient, Eq. (5)

# --- CAST hyper-parameters ---------------------------------------------------------
ALPHA_D = 0.7                # routing weight on normalised distance, Eq. (2)
ALPHA_C = 0.3                # routing weight on congestion penalty, Eq. (2)
EPS_RHO = 1e-6               # epsilon in Psi(rho) = rho / (1 - rho + eps), Eq. (10)
ETA = (0.30, 0.25, 0.25, 0.20)   # edge-score weights eta_1..4 (traffic, loadbal, spectral, dist)
DELTA_SCORE = 0.05           # score threshold delta, Sec. IV (Algorithm 1)
MAX_ITER = 10                # K, alternating-optimisation iterations
NODE_THRESHOLD_LANCZOS = 500  # sub-graph size below which exact Lanczos is used

# --- Access segment -----------------------------------------------------------------
MIN_ELEVATION_DEG = 10.0     # ground-station mask, Sec. III-D
FLOW_RATE_MBPS = 40.0        # per-flow peak rate, Sec. IV-A

# --- Queueing model (analytic mode) -------------------------------------------------
# The manuscript reports path-total T_queue, which includes, per the author-provided
# note 02_queueing_model.md / 16_fact_confirmations.md, both FIFO waiting and a
# slotted/framing wait (an effective per-hop service quantum of ~0.35 ms rather than
# the 0.586 us raw packet transmission time).  We expose that quantum explicitly so
# the assumption is visible rather than hidden.
QUEUE_FRAME_QUANTUM_S = 0.35e-3
TAIL_PERCENTILE = 95.0
