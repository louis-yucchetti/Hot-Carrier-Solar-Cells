from __future__ import annotations

import numpy as np

# ----------------------------- User-tunable settings -----------------------------
FILENAME = "GaAs bulk_PL_avg_circle_4pixs.txt"

# If False, fixed fit window below is used for all spectra.
AUTO_SELECT_FIT_WINDOW = True
FIT_ENERGY_MIN_EV = 1.55
FIT_ENERGY_MAX_EV = 1.70

# Auto-window selection parameters
WINDOW_SEARCH_MIN_EV = 1.45
WINDOW_SEARCH_MAX_EV = 1.77
WINDOW_PEAK_OFFSET_EV = 0.0
WINDOW_MIN_POINTS = 18
WINDOW_MIN_R2 = 0.995
WINDOW_T_MIN_K = 150.0
WINDOW_T_MAX_K = 1200.0

# Fit-range uncertainty from an objective ensemble of plausible windows
ESTIMATE_FIT_RANGE_UNCERTAINTY = True
FIT_RANGE_SCAN_MIN_POINTS = 12
FIT_RANGE_SCAN_MIN_R2 = 0.99
FIT_RANGE_SCAN_REQUIRE_PHYSICAL_BOUNDS = True
FIT_RANGE_SCAN_MAX_WINDOWS_TO_PLOT = 180
FIT_RANGE_SCAN_PLOT_WEIGHT_COVERAGE = 0.95

# High-energy absorptivity A0 from OptiPV simulation in [1.50, 1.70] eV.
A0_HIGH_ENERGY_MIN = 0.459
A0_HIGH_ENERGY_MAX = 0.555
A0_UNCERTAINTY_MODEL = "uniform"  # "uniform" or "half_range"

if A0_UNCERTAINTY_MODEL == "uniform":
    A0_SIGMA = (A0_HIGH_ENERGY_MAX - A0_HIGH_ENERGY_MIN) / np.sqrt(12.0)
elif A0_UNCERTAINTY_MODEL == "half_range":
    A0_SIGMA = 0.5 * (A0_HIGH_ENERGY_MAX - A0_HIGH_ENERGY_MIN)
else:
    raise ValueError("A0_UNCERTAINTY_MODEL must be 'uniform' or 'half_range'.")

# Figure export/style
SAVE_DPI = 450

# Nominal A0 used to convert Delta_mu_eff into Delta_mu.
ASSUMED_A0 = 0.5 * (A0_HIGH_ENERGY_MIN + A0_HIGH_ENERGY_MAX)

# Laser/power-balance model inputs
LASER_WAVELENGTH_NM = 532.0
ABSORPTIVITY_AT_LASER = 0.6
ABSORPTIVITY_AT_LASER_SIGMA = 0.0
PLQY_ETA = 0.0
PLQY_ETA_SIGMA = 0.0
ACTIVE_LAYER_THICKNESS_NM = 950.0

# Optional Tsai-model lookup table for direct experiment/theory comparison.
# CSV columns required: n_cm3, temperature_k, p_th_w_cm3
TSAI_MODEL_TABLE_CSV = ""

# GaAs parameters for carrier-statistics post-processing (MB and FD)
EG_EV = 1.424
M_E_EFF = 0.067
M_H_EFF = 0.50

# Fermi-Dirac solver controls (3D parabolic-band model)
FD_BISECTION_ETA_BOUND = 40.0
FD_BISECTION_MAX_ITER = 160
FD_BISECTION_TOL = 1e-8
FD_F12_MIDPOINTS = 1200

# Excitation intensities (W/cm^2), one per spectrum column
EXCITATION_INTENSITY_W_CM2 = np.array(
    [
        9.45839419e01,
        1.33106541e02,
        1.72255523e02,
        2.55564558e02,
        4.09028570e02,
        6.10097744e02,
        9.18278535e02,
        1.24274530e03,
        1.86662349e03,
        2.38025814e03,
        3.78335768e03,
        7.71715161e03,
        1.27500766e04,
        2.47332312e04,
        4.25641654e04,
        5.73274119e04,
        7.13237366e04,
        8.42655436e04,
        9.67280245e04,
        1.08615314e05,
        1.19639816e05,
        1.29993262e05,
    ],
    dtype=float,
)

# ----------------------------- Physical constants -------------------------------
H = 6.62607015e-34  # J.s
C = 2.99792458e8  # m/s
K_B = 1.380649e-23  # J/K
E_CHARGE = 1.602176634e-19  # J/eV
HBAR = 1.054571817e-34  # J.s
M0 = 9.1093837015e-31  # kg
