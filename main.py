from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.tri import Triangulation
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter


# ----------------------------- User-tunable settings -----------------------------
FILENAME = "GaAs bulk_PL_avg_circle_4pixs.txt"

# If False, fixed fit window below is used for all spectra.
AUTO_SELECT_FIT_WINDOW = True
FIT_ENERGY_MIN_EV = 1.55
FIT_ENERGY_MAX_EV = 1.70

# Auto-window selection parameters
WINDOW_SEARCH_MIN_EV = 1.45
WINDOW_SEARCH_MAX_EV = 1.75
WINDOW_PEAK_OFFSET_EV = 0.045
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
    raise ValueError(
        "A0_UNCERTAINTY_MODEL must be 'uniform' or 'half_range'."
    )

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


@dataclass
class FitResult:
    spectrum_id: str
    intensity_w_cm2: float
    a0_value: float
    a0_sigma: float
    fit_min_ev: float
    fit_max_ev: float
    window_mode: str
    n_points_fit: int
    slope: float
    intercept: float
    r2: float
    temperature_k: float
    qfls_effective_ev: float
    qfls_ev: float
    mu_e_ev: float
    mu_h_ev: float
    carrier_density_cm3: float
    mu_e_fd_ev: float
    mu_h_fd_ev: float
    carrier_density_fd_cm3: float
    temperature_err_chi2_k: float
    temperature_err_range_k: float
    temperature_err_a0_k: float
    temperature_err_total_k: float
    qfls_effective_err_chi2_ev: float
    qfls_effective_err_range_ev: float
    qfls_effective_err_a0_ev: float
    qfls_effective_err_total_ev: float
    qfls_err_chi2_ev: float
    qfls_err_range_ev: float
    qfls_err_a0_ev: float
    qfls_err_total_ev: float
    mu_e_err_chi2_ev: float
    mu_e_err_range_ev: float
    mu_e_err_a0_ev: float
    mu_e_err_total_ev: float
    mu_h_err_chi2_ev: float
    mu_h_err_range_ev: float
    mu_h_err_a0_ev: float
    mu_h_err_total_ev: float
    carrier_density_err_chi2_cm3: float
    carrier_density_err_range_cm3: float
    carrier_density_err_a0_cm3: float
    carrier_density_err_total_cm3: float
    fit_range_samples: int


@dataclass
class WindowFitSample:
    idx_start: int
    idx_end: int
    n_points: int
    fit_min_ev: float
    fit_max_ev: float
    r2: float
    aicc: float
    temperature_k: float
    qfls_effective_ev: float
    qfls_ev: float
    mu_e_ev: float
    mu_h_ev: float
    carrier_density_cm3: float


FIT_PARAMETER_KEYS = (
    "temperature_k",
    "qfls_effective_ev",
    "qfls_ev",
    "mu_e_ev",
    "mu_h_ev",
    "carrier_density_cm3",
)

FIT_RESULT_ERROR_FIELDS = {
    "temperature_k": (
        "temperature_err_chi2_k",
        "temperature_err_range_k",
        "temperature_err_a0_k",
        "temperature_err_total_k",
    ),
    "qfls_effective_ev": (
        "qfls_effective_err_chi2_ev",
        "qfls_effective_err_range_ev",
        "qfls_effective_err_a0_ev",
        "qfls_effective_err_total_ev",
    ),
    "qfls_ev": (
        "qfls_err_chi2_ev",
        "qfls_err_range_ev",
        "qfls_err_a0_ev",
        "qfls_err_total_ev",
    ),
    "mu_e_ev": (
        "mu_e_err_chi2_ev",
        "mu_e_err_range_ev",
        "mu_e_err_a0_ev",
        "mu_e_err_total_ev",
    ),
    "mu_h_ev": (
        "mu_h_err_chi2_ev",
        "mu_h_err_range_ev",
        "mu_h_err_a0_ev",
        "mu_h_err_total_ev",
    ),
    "carrier_density_cm3": (
        "carrier_density_err_chi2_cm3",
        "carrier_density_err_range_cm3",
        "carrier_density_err_a0_cm3",
        "carrier_density_err_total_cm3",
    ),
}


def _empty_like_fit_result(
    *,
    spectrum_id: str,
    intensity_w_cm2: float,
    a0_value: float,
    a0_sigma: float,
    fit_min_ev: float,
    fit_max_ev: float,
    window_mode: str,
    n_points_fit: int,
) -> dict[str, float | str | int]:
    values: dict[str, float | str | int] = {field.name: np.nan for field in fields(FitResult)}
    values.update(
        {
            "spectrum_id": spectrum_id,
            "intensity_w_cm2": float(intensity_w_cm2),
            "a0_value": float(a0_value),
            "a0_sigma": float(a0_sigma),
            "fit_min_ev": float(fit_min_ev),
            "fit_max_ev": float(fit_max_ev),
            "window_mode": window_mode,
            "n_points_fit": int(n_points_fit),
            "fit_range_samples": 0,
        }
    )
    return values


def _combine_parameter_error_components(
    chi2_err: dict[str, float],
    range_err: dict[str, float],
    a0_err: dict[str, float],
) -> dict[str, float]:
    return {
        key: _combine_uncertainties(chi2_err[key], range_err[key], a0_err[key])
        for key in FIT_PARAMETER_KEYS
    }


def _build_intensity_model(
    energy_ev: np.ndarray,
    temperature_k: float,
    qfls_effective_ev: float,
) -> np.ndarray:
    if (
        (not np.isfinite(temperature_k))
        or (temperature_k <= 0)
        or (not np.isfinite(qfls_effective_ev))
    ):
        return np.full_like(energy_ev, np.nan, dtype=float)

    qfls_effective_j = qfls_effective_ev * E_CHARGE
    energy_j_all = energy_ev * E_CHARGE
    ln_prefactor = np.log(2.0 * energy_j_all**2 / (H**3 * C**2))
    ln_i_model = ln_prefactor - (energy_j_all - qfls_effective_j) / (K_B * temperature_k)
    return np.exp(ln_i_model)


def _build_fit_result(
    *,
    spectrum_id: str,
    intensity_w_cm2: float,
    a0_value: float,
    a0_sigma: float,
    fit_min_ev: float,
    fit_max_ev: float,
    window_mode: str,
    n_points_fit: int,
    slope: float,
    intercept: float,
    r2: float,
    base_parameters: dict[str, float],
    mu_e_fd_ev: float,
    mu_h_fd_ev: float,
    carrier_density_fd_cm3: float,
    chi2_err: dict[str, float],
    range_err: dict[str, float],
    a0_err: dict[str, float],
    total_err: dict[str, float],
    fit_range_samples: int,
) -> FitResult:
    values = _empty_like_fit_result(
        spectrum_id=spectrum_id,
        intensity_w_cm2=intensity_w_cm2,
        a0_value=a0_value,
        a0_sigma=a0_sigma,
        fit_min_ev=fit_min_ev,
        fit_max_ev=fit_max_ev,
        window_mode=window_mode,
        n_points_fit=n_points_fit,
    )
    values.update(
        {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r2),
            "temperature_k": float(base_parameters["temperature_k"]),
            "qfls_effective_ev": float(base_parameters["qfls_effective_ev"]),
            "qfls_ev": float(base_parameters["qfls_ev"]),
            "mu_e_ev": float(base_parameters["mu_e_ev"]),
            "mu_h_ev": float(base_parameters["mu_h_ev"]),
            "carrier_density_cm3": float(base_parameters["carrier_density_cm3"]),
            "mu_e_fd_ev": float(mu_e_fd_ev),
            "mu_h_fd_ev": float(mu_h_fd_ev),
            "carrier_density_fd_cm3": float(carrier_density_fd_cm3),
            "fit_range_samples": int(fit_range_samples),
        }
    )
    for key, (
        chi2_field,
        range_field,
        a0_field,
        total_field,
    ) in FIT_RESULT_ERROR_FIELDS.items():
        values[chi2_field] = float(chi2_err[key])
        values[range_field] = float(range_err[key])
        values[a0_field] = float(a0_err[key])
        values[total_field] = float(total_err[key])
    return FitResult(**values)


def setup_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 5.0,
            "xtick.minor.size": 2.8,
            "ytick.major.size": 5.0,
            "ytick.minor.size": 2.8,
            "legend.frameon": True,
            "legend.framealpha": 0.93,
            "legend.fancybox": False,
            "legend.edgecolor": "0.25",
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "lines.linewidth": 1.7,
            "savefig.dpi": SAVE_DPI,
        }
    )


def style_axes(ax: plt.Axes, logx: bool = False, logy: bool = False) -> None:
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if not logx:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    if not logy:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if logy:
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", linewidth=0.7)
    ax.grid(True, which="minor", linewidth=0.35, alpha=0.12)


def save_figure(fig: plt.Figure, outpath: Path) -> None:
    png_outpath = outpath.with_suffix(".png")
    fig.savefig(png_outpath, dpi=SAVE_DPI, bbox_inches="tight")


def load_spectra(data_dir: Path, filename: str) -> pd.DataFrame:
    file_path = data_dir.resolve() / filename
    return pd.read_csv(file_path, sep=";", index_col=0)


def linearized_signal(energy_ev: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    energy_j = energy_ev * E_CHARGE
    return np.log((H**3 * C**2 / (2.0 * energy_j**2)) * intensity)


def _effective_density_of_states(
    temperature_k: float,
    m_e_eff: float = M_E_EFF,
    m_h_eff: float = M_H_EFF,
) -> tuple[float, float]:
    if (not np.isfinite(temperature_k)) or temperature_k <= 0:
        return np.nan, np.nan

    nc = 2.0 * ((m_e_eff * M0 * K_B * temperature_k) / (2.0 * np.pi * HBAR**2)) ** 1.5
    nv = 2.0 * ((m_h_eff * M0 * K_B * temperature_k) / (2.0 * np.pi * HBAR**2)) ** 1.5
    return float(nc), float(nv)


def compute_mu_and_density_mb(
    temperature_k: float,
    qfls_ev: float,
    eg_ev: float = EG_EV,
    m_e_eff: float = M_E_EFF,
    m_h_eff: float = M_H_EFF,
) -> tuple[float, float, float]:
    nc, nv = _effective_density_of_states(
        temperature_k=temperature_k,
        m_e_eff=m_e_eff,
        m_h_eff=m_h_eff,
    )
    if (not np.isfinite(nc)) or (not np.isfinite(nv)) or (nc <= 0) or (nv <= 0):
        return np.nan, np.nan, np.nan

    delta_mass_term_ev = (K_B * temperature_k / E_CHARGE) * np.log(nc / nv)
    mu_e_ev = 0.5 * (qfls_ev - delta_mass_term_ev)
    mu_h_ev = 0.5 * (qfls_ev + delta_mass_term_ev)

    n_m3 = nc * np.exp(((mu_e_ev - eg_ev / 2.0) * E_CHARGE) / (K_B * temperature_k))
    n_cm3 = n_m3 / 1e6
    return mu_e_ev, mu_h_ev, n_cm3


def _fermi_dirac_half(eta: float) -> float:
    """
    Complete Fermi-Dirac integral F_{1/2}(eta):
      F_{1/2}(eta) = (2/sqrt(pi)) * integral_0^inf sqrt(eps)/(1 + exp(eps-eta)) d eps
    """
    if not np.isfinite(eta):
        return np.nan

    eta = float(eta)
    if eta <= -8.0:
        return float(np.exp(eta))

    if eta >= 12.0:
        leading = (4.0 / (3.0 * np.sqrt(np.pi))) * eta**1.5
        correction = (np.pi**2 / (6.0 * np.sqrt(np.pi))) * eta**-0.5
        return float(leading + correction)

    x_max = max(40.0, eta + 40.0)
    x = np.linspace(0.0, x_max, FD_F12_MIDPOINTS, dtype=float)
    u = x - eta
    occupation = np.where(u > 40.0, np.exp(-u), 1.0 / (1.0 + np.exp(u)))
    integrand = np.sqrt(x) * occupation
    return float((2.0 / np.sqrt(np.pi)) * np.trapezoid(integrand, x))


def compute_mu_and_density_fd(
    temperature_k: float,
    qfls_ev: float,
    eg_ev: float = EG_EV,
    m_e_eff: float = M_E_EFF,
    m_h_eff: float = M_H_EFF,
) -> tuple[float, float, float]:
    if (
        (not np.isfinite(temperature_k))
        or (temperature_k <= 0)
        or (not np.isfinite(qfls_ev))
    ):
        return np.nan, np.nan, np.nan

    nc, nv = _effective_density_of_states(
        temperature_k=temperature_k,
        m_e_eff=m_e_eff,
        m_h_eff=m_h_eff,
    )
    if (not np.isfinite(nc)) or (not np.isfinite(nv)) or (nc <= 0) or (nv <= 0):
        return np.nan, np.nan, np.nan

    kbt_ev = (K_B * temperature_k) / E_CHARGE
    reduced_qfls = (qfls_ev - eg_ev) / kbt_ev

    def neutrality(eta_e: float) -> float:
        eta_h = reduced_qfls - eta_e
        return (nc * _fermi_dirac_half(eta_e)) - (nv * _fermi_dirac_half(eta_h))

    eta_lo = min(-FD_BISECTION_ETA_BOUND, reduced_qfls - FD_BISECTION_ETA_BOUND)
    eta_hi = max(FD_BISECTION_ETA_BOUND, reduced_qfls + FD_BISECTION_ETA_BOUND)
    f_lo = neutrality(eta_lo)
    f_hi = neutrality(eta_hi)
    if (not np.isfinite(f_lo)) or (not np.isfinite(f_hi)):
        return np.nan, np.nan, np.nan

    while f_lo * f_hi > 0:
        eta_lo -= FD_BISECTION_ETA_BOUND
        eta_hi += FD_BISECTION_ETA_BOUND
        f_lo = neutrality(eta_lo)
        f_hi = neutrality(eta_hi)
        if (not np.isfinite(f_lo)) or (not np.isfinite(f_hi)):
            return np.nan, np.nan, np.nan
        if max(abs(eta_lo), abs(eta_hi)) > 8.0 * FD_BISECTION_ETA_BOUND:
            return np.nan, np.nan, np.nan

    eta_mid = 0.5 * (eta_lo + eta_hi)
    for _ in range(FD_BISECTION_MAX_ITER):
        eta_mid = 0.5 * (eta_lo + eta_hi)
        f_mid = neutrality(eta_mid)
        if not np.isfinite(f_mid):
            return np.nan, np.nan, np.nan

        if abs(eta_hi - eta_lo) < FD_BISECTION_TOL:
            break
        if f_lo * f_mid <= 0:
            eta_hi = eta_mid
            f_hi = f_mid
        else:
            eta_lo = eta_mid
            f_lo = f_mid

    eta_e = float(eta_mid)
    eta_h = float(reduced_qfls - eta_e)
    mu_e_ev = eg_ev / 2.0 + kbt_ev * eta_e
    mu_h_ev = eg_ev / 2.0 + kbt_ev * eta_h
    n_m3 = nc * _fermi_dirac_half(eta_e)
    n_cm3 = n_m3 / 1e6
    return float(mu_e_ev), float(mu_h_ev), float(n_cm3)


def _compute_linear_fit_and_covariance(
    x_j: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, np.ndarray, float]:
    slope, intercept = np.polyfit(x_j, y, deg=1)
    y_fit = slope * x_j + intercept
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    covariance = np.full((2, 2), np.nan, dtype=float)
    n_points = int(x_j.size)
    if n_points > 2:
        x_mean = float(np.mean(x_j))
        sxx = float(np.sum((x_j - x_mean) ** 2))
        if sxx > 0 and np.isfinite(ss_res):
            sigma2 = ss_res / (n_points - 2)
            covariance[0, 0] = sigma2 / sxx
            covariance[1, 1] = sigma2 * (1.0 / n_points + x_mean**2 / sxx)
            covariance[0, 1] = covariance[1, 0] = -x_mean * sigma2 / sxx

    return float(slope), float(intercept), float(r2), covariance, ss_res


def _compute_parameters_from_line(
    slope: float, intercept: float, assumed_a0: float
) -> tuple[float, float, float, float, float, float]:
    if (not np.isfinite(slope)) or slope == 0 or (not np.isfinite(intercept)):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    temperature_k = -1.0 / (K_B * slope)
    qfls_effective_j = intercept * K_B * temperature_k
    qfls_effective_ev = qfls_effective_j / E_CHARGE
    qfls_ev = qfls_effective_ev - (K_B * temperature_k / E_CHARGE) * np.log(assumed_a0)
    mu_e_ev, mu_h_ev, n_cm3 = compute_mu_and_density_mb(
        temperature_k=temperature_k,
        qfls_ev=qfls_ev,
    )
    return (
        float(temperature_k),
        float(qfls_effective_ev),
        float(qfls_ev),
        float(mu_e_ev),
        float(mu_h_ev),
        float(n_cm3),
    )


def _jacobian_mu_density_wrt_t_q(temperature_k: float, qfls_ev: float) -> np.ndarray:
    jac = np.full((3, 2), np.nan, dtype=float)
    if (not np.isfinite(temperature_k)) or (not np.isfinite(qfls_ev)) or temperature_k <= 0:
        return jac

    dt = max(1e-3, 1e-4 * abs(temperature_k))
    t_minus = max(1e-6, temperature_k - dt)
    t_plus = temperature_k + dt
    dt_eff = t_plus - t_minus

    dq = max(1e-6, 1e-4 * abs(qfls_ev))
    q_minus = qfls_ev - dq
    q_plus = qfls_ev + dq
    dq_eff = q_plus - q_minus

    mu_e_tp, mu_h_tp, n_tp = compute_mu_and_density_mb(temperature_k=t_plus, qfls_ev=qfls_ev)
    mu_e_tm, mu_h_tm, n_tm = compute_mu_and_density_mb(temperature_k=t_minus, qfls_ev=qfls_ev)
    mu_e_qp, mu_h_qp, n_qp = compute_mu_and_density_mb(temperature_k=temperature_k, qfls_ev=q_plus)
    mu_e_qm, mu_h_qm, n_qm = compute_mu_and_density_mb(temperature_k=temperature_k, qfls_ev=q_minus)

    jac[:, 0] = np.array(
        [
            (mu_e_tp - mu_e_tm) / dt_eff,
            (mu_h_tp - mu_h_tm) / dt_eff,
            (n_tp - n_tm) / dt_eff,
        ],
        dtype=float,
    )
    jac[:, 1] = np.array(
        [
            (mu_e_qp - mu_e_qm) / dq_eff,
            (mu_h_qp - mu_h_qm) / dq_eff,
            (n_qp - n_qm) / dq_eff,
        ],
        dtype=float,
    )
    return jac


def _chi2_parameter_uncertainties(
    slope: float,
    intercept: float,
    covariance_line: np.ndarray,
    temperature_k: float,
    qfls_ev: float,
    assumed_a0: float,
) -> dict[str, float]:
    out = {
        "temperature_k": np.nan,
        "qfls_effective_ev": np.nan,
        "qfls_ev": np.nan,
        "mu_e_ev": np.nan,
        "mu_h_ev": np.nan,
        "carrier_density_cm3": np.nan,
    }
    if (not np.isfinite(slope)) or slope == 0:
        return out
    if covariance_line.shape != (2, 2) or not np.all(np.isfinite(covariance_line)):
        return out

    d_t_dm = 1.0 / (K_B * slope**2)
    d_qeff_dm = intercept / (slope**2 * E_CHARGE)
    d_qeff_db = -1.0 / (slope * E_CHARGE)
    alpha = (K_B / E_CHARGE) * np.log(assumed_a0)
    d_q_dm = d_qeff_dm - alpha * d_t_dm
    d_q_db = d_qeff_db

    jac_primary = np.array(
        [
            [d_t_dm, 0.0],
            [d_qeff_dm, d_qeff_db],
            [d_q_dm, d_q_db],
        ],
        dtype=float,
    )
    cov_primary = jac_primary @ covariance_line @ jac_primary.T
    var_t = cov_primary[0, 0]
    var_qeff = cov_primary[1, 1]
    var_q = cov_primary[2, 2]

    out["temperature_k"] = float(np.sqrt(max(var_t, 0.0)))
    out["qfls_effective_ev"] = float(np.sqrt(max(var_qeff, 0.0)))
    out["qfls_ev"] = float(np.sqrt(max(var_q, 0.0)))

    cov_tq = cov_primary[np.ix_([0, 2], [0, 2])]
    jac_derived = _jacobian_mu_density_wrt_t_q(temperature_k=temperature_k, qfls_ev=qfls_ev)
    if np.all(np.isfinite(jac_derived)) and np.all(np.isfinite(cov_tq)):
        cov_derived = jac_derived @ cov_tq @ jac_derived.T
        out["mu_e_ev"] = float(np.sqrt(max(cov_derived[0, 0], 0.0)))
        out["mu_h_ev"] = float(np.sqrt(max(cov_derived[1, 1], 0.0)))
        out["carrier_density_cm3"] = float(np.sqrt(max(cov_derived[2, 2], 0.0)))

    return out


def _build_scan_candidate_mask(
    energy_ev: np.ndarray,
    intensity: np.ndarray,
    min_points: int,
    prefer_peak_offset: bool,
) -> tuple[np.ndarray, str]:
    valid = np.isfinite(energy_ev) & np.isfinite(intensity) & (intensity > 0)
    if np.count_nonzero(valid) < min_points:
        return valid, "fallback_insufficient_points"

    if prefer_peak_offset:
        peak_idx = np.argmax(np.where(valid, intensity, -np.inf))
        peak_ev = float(energy_ev[peak_idx])
        search_min = max(WINDOW_SEARCH_MIN_EV, peak_ev + WINDOW_PEAK_OFFSET_EV)
        candidate = (
            valid
            & (energy_ev >= search_min)
            & (energy_ev <= WINDOW_SEARCH_MAX_EV)
        )
        mode = "auto_peak_offset"
        if np.count_nonzero(candidate) >= min_points:
            return candidate, mode

        candidate = (
            valid
            & (energy_ev >= WINDOW_SEARCH_MIN_EV)
            & (energy_ev <= WINDOW_SEARCH_MAX_EV)
        )
        mode = "fallback_search_range_only"
    else:
        candidate = (
            valid
            & (energy_ev >= WINDOW_SEARCH_MIN_EV)
            & (energy_ev <= WINDOW_SEARCH_MAX_EV)
        )
        mode = "search_range_only"

    if np.count_nonzero(candidate) < min_points:
        candidate = valid
        mode += "|fallback_all_valid"
    return candidate, mode


def _enumerate_window_fit_samples(
    energy_ev: np.ndarray,
    intensity: np.ndarray,
    candidate_mask: np.ndarray,
    min_points: int,
    min_r2: float,
    require_physical_bounds: bool,
    assumed_a0: float,
) -> list[WindowFitSample]:
    idx_candidate = np.flatnonzero(candidate_mask)
    if idx_candidate.size < min_points:
        return []

    energy_j = energy_ev * E_CHARGE
    valid_points = np.isfinite(energy_ev) & np.isfinite(intensity) & (intensity > 0)
    y_linearized = np.full_like(energy_ev, np.nan, dtype=float)
    y_linearized[valid_points] = linearized_signal(
        energy_ev[valid_points], intensity[valid_points]
    )

    energy_j_candidate = energy_j[idx_candidate]
    y_candidate = y_linearized[idx_candidate]

    samples: list[WindowFitSample] = []
    n_candidate = idx_candidate.size

    for i in range(0, n_candidate - min_points + 1):
        for j in range(i + min_points - 1, n_candidate):
            idx_window = idx_candidate[i : j + 1]
            x_j = energy_j_candidate[i : j + 1]
            y = y_candidate[i : j + 1]
            if not np.all(np.isfinite(y)):
                continue
            slope, intercept, r2, _, ss_res = _compute_linear_fit_and_covariance(
                x_j, y
            )
            if (not np.isfinite(slope)) or (slope >= 0):
                continue
            if (not np.isfinite(r2)) or (r2 < min_r2):
                continue

            (
                temperature_k,
                qfls_effective_ev,
                qfls_ev,
                mu_e_ev,
                mu_h_ev,
                n_cm3,
            ) = _compute_parameters_from_line(
                slope=slope,
                intercept=intercept,
                assumed_a0=assumed_a0,
            )
            if require_physical_bounds and (
                (not np.isfinite(temperature_k))
                or (temperature_k < WINDOW_T_MIN_K)
                or (temperature_k > WINDOW_T_MAX_K)
            ):
                continue

            if not np.all(
                np.isfinite(
                    np.array(
                        [
                            temperature_k,
                            qfls_effective_ev,
                            qfls_ev,
                            mu_e_ev,
                            mu_h_ev,
                            n_cm3,
                        ],
                        dtype=float,
                    )
                )
            ):
                continue

            aicc = _compute_aicc(
                ss_res=ss_res,
                n_points=idx_window.size,
                n_parameters=2,
            )
            if not np.isfinite(aicc):
                continue

            samples.append(
                WindowFitSample(
                    idx_start=int(idx_window[0]),
                    idx_end=int(idx_window[-1]),
                    n_points=int(idx_window.size),
                    fit_min_ev=float(energy_ev[idx_window[0]]),
                    fit_max_ev=float(energy_ev[idx_window[-1]]),
                    r2=float(r2),
                    aicc=float(aicc),
                    temperature_k=float(temperature_k),
                    qfls_effective_ev=float(qfls_effective_ev),
                    qfls_ev=float(qfls_ev),
                    mu_e_ev=float(mu_e_ev),
                    mu_h_ev=float(mu_h_ev),
                    carrier_density_cm3=float(n_cm3),
                )
            )

    return samples


def _fit_range_rms_uncertainty(
    energy_ev: np.ndarray,
    intensity: np.ndarray,
    base_fit_mask: np.ndarray,
    base_parameters: dict[str, float],
    assumed_a0: float,
    scan_candidate_mask: np.ndarray | None = None,
) -> tuple[
    dict[str, float],
    int,
    list[tuple[float, float]],
    tuple[float, float] | None,
]:
    out = {key: np.nan for key in base_parameters}
    min_points = max(3, FIT_RANGE_SCAN_MIN_POINTS)
    valid = np.isfinite(energy_ev) & np.isfinite(intensity) & (intensity > 0)

    if scan_candidate_mask is None:
        candidate = (
            valid
            & (energy_ev >= WINDOW_SEARCH_MIN_EV)
            & (energy_ev <= WINDOW_SEARCH_MAX_EV)
        )
    else:
        candidate = valid & scan_candidate_mask

    if np.count_nonzero(candidate) < min_points:
        candidate = valid & base_fit_mask
    if np.count_nonzero(candidate) < min_points:
        candidate = valid

    idx_candidate = np.flatnonzero(candidate)
    if idx_candidate.size == 0:
        return out, 0, [], None

    scan_bounds = (
        float(energy_ev[idx_candidate[0]]),
        float(energy_ev[idx_candidate[-1]]),
    )
    if not ESTIMATE_FIT_RANGE_UNCERTAINTY:
        return out, 0, [], scan_bounds

    samples = _enumerate_window_fit_samples(
        energy_ev=energy_ev,
        intensity=intensity,
        candidate_mask=candidate,
        min_points=min_points,
        min_r2=FIT_RANGE_SCAN_MIN_R2,
        require_physical_bounds=FIT_RANGE_SCAN_REQUIRE_PHYSICAL_BOUNDS,
        assumed_a0=assumed_a0,
    )
    if not samples:
        return out, 0, [], scan_bounds

    aicc = np.array([sample.aicc for sample in samples], dtype=float)
    delta_aicc = aicc - float(np.min(aicc))
    weights = np.exp(-0.5 * delta_aicc)
    if (not np.all(np.isfinite(weights))) or (np.sum(weights) <= 0):
        weights = np.ones_like(weights, dtype=float)
    weights = weights / np.sum(weights)

    for key, base_value in base_parameters.items():
        if not np.isfinite(base_value):
            out[key] = np.nan
            continue

        sample_values = np.array([getattr(sample, key) for sample in samples], dtype=float)
        finite = np.isfinite(sample_values)
        if not np.any(finite):
            out[key] = np.nan
            continue

        values = sample_values[finite]
        w = weights[finite]
        w_sum = np.sum(w)
        if w_sum <= 0:
            out[key] = np.nan
            continue

        w = w / w_sum
        out[key] = float(np.sqrt(np.sum(w * (values - base_value) ** 2)))

    plot_windows: list[tuple[float, float]] = []
    if FIT_RANGE_SCAN_MAX_WINDOWS_TO_PLOT > 0:
        order = np.argsort(weights)[::-1]
        weight_covered = 0.0
        for idx in order:
            plot_windows.append((samples[idx].fit_min_ev, samples[idx].fit_max_ev))
            weight_covered += float(weights[idx])
            if (
                len(plot_windows) >= FIT_RANGE_SCAN_MAX_WINDOWS_TO_PLOT
                or weight_covered >= FIT_RANGE_SCAN_PLOT_WEIGHT_COVERAGE
            ):
                break

    return out, len(samples), plot_windows, scan_bounds


def _compute_aicc(ss_res: float, n_points: int, n_parameters: int = 2) -> float:
    if (not np.isfinite(ss_res)) or (n_points <= n_parameters + 1):
        return np.nan

    rss = max(float(ss_res), np.finfo(float).tiny)
    aic = n_points * np.log(rss / n_points) + 2.0 * n_parameters
    correction = (2.0 * n_parameters * (n_parameters + 1)) / (
        n_points - n_parameters - 1
    )
    return float(aic + correction)


def _a0_parameter_uncertainties(
    temperature_k: float,
    qfls_ev: float,
    assumed_a0: float,
    a0_sigma: float,
) -> dict[str, float]:
    out = {
        "temperature_k": 0.0,
        "qfls_effective_ev": 0.0,
        "qfls_ev": 0.0,
        "mu_e_ev": 0.0,
        "mu_h_ev": 0.0,
        "carrier_density_cm3": 0.0,
    }
    if (
        (not np.isfinite(temperature_k))
        or (temperature_k <= 0)
        or (not np.isfinite(qfls_ev))
        or (not np.isfinite(assumed_a0))
        or (assumed_a0 <= 0)
    ):
        out["qfls_ev"] = np.nan
        out["mu_e_ev"] = np.nan
        out["mu_h_ev"] = np.nan
        out["carrier_density_cm3"] = np.nan
        return out

    if (not np.isfinite(a0_sigma)) or (a0_sigma <= 0):
        return out

    d_q_d_a0 = -(K_B * temperature_k) / (E_CHARGE * assumed_a0)
    sigma_q = abs(d_q_d_a0) * abs(a0_sigma)
    out["qfls_ev"] = float(sigma_q)

    cov_tq = np.array([[0.0, 0.0], [0.0, sigma_q**2]], dtype=float)
    jac_derived = _jacobian_mu_density_wrt_t_q(
        temperature_k=temperature_k, qfls_ev=qfls_ev
    )
    if np.all(np.isfinite(jac_derived)):
        cov_derived = jac_derived @ cov_tq @ jac_derived.T
        out["mu_e_ev"] = float(np.sqrt(max(cov_derived[0, 0], 0.0)))
        out["mu_h_ev"] = float(np.sqrt(max(cov_derived[1, 1], 0.0)))
        out["carrier_density_cm3"] = float(np.sqrt(max(cov_derived[2, 2], 0.0)))

    return out


def _combine_uncertainties(*errors: float) -> float:
    finite_errors = np.array(
        [float(abs(err)) for err in errors if np.isfinite(err)], dtype=float
    )
    if finite_errors.size == 0:
        return np.nan
    return float(np.sqrt(np.sum(finite_errors**2)))


def _sanitize_nonnegative(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.where(np.isfinite(arr) & (arr >= 0), arr, 0.0)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    return np.divide(
        num,
        den,
        out=np.full_like(num, np.nan, dtype=float),
        where=np.isfinite(num) & np.isfinite(den) & (den > 0),
    )


def _assign_dataframe_columns(
    dataframe: pd.DataFrame,
    values: dict[str, float | np.ndarray],
) -> None:
    for column, value in values.items():
        dataframe[column] = value


def _safe_log_yerr(y: np.ndarray, err: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    err = np.asarray(err, dtype=float)
    lower = np.where(np.isfinite(y) & np.isfinite(err) & (y > 0), np.minimum(err, 0.95 * y), np.nan)
    upper = np.where(np.isfinite(err), err, np.nan)
    return np.vstack([lower, upper])


def _laser_photon_energy_from_wavelength(wavelength_nm: float) -> tuple[float, float]:
    wavelength_m = wavelength_nm * 1e-9
    energy_j = (H * C) / wavelength_m
    energy_ev = energy_j / E_CHARGE
    return float(energy_j), float(energy_ev)


def compute_power_balance_table(
    results_df: pd.DataFrame,
    laser_wavelength_nm: float = LASER_WAVELENGTH_NM,
    absorptivity_at_laser: float = ABSORPTIVITY_AT_LASER,
    absorptivity_at_laser_sigma: float = ABSORPTIVITY_AT_LASER_SIGMA,
    plqy_eta: float = PLQY_ETA,
    plqy_eta_sigma: float = PLQY_ETA_SIGMA,
    active_layer_thickness_nm: float = ACTIVE_LAYER_THICKNESS_NM,
    eg_ev: float = EG_EV,
) -> pd.DataFrame:
    """
    Add power-balance quantities per spectrum using:
      P_abs = A_laser * P_exc
      phi_abs = P_abs / E_laser
      phi_rad = eta * phi_abs
      phi_nonrad = (1 - eta) * phi_abs
      P_rec = phi_nonrad*(Eg + 3kBT) + phi_rad*(Eg + kBT)
      P_th = P_abs - P_rec
    Then convert area-based powers (W cm^-2) to volumetric powers (W cm^-3)
    using the active-layer thickness.
    """
    df = results_df.copy()
    p_exc_w_cm2 = df["intensity_w_cm2"].to_numpy(dtype=float)
    temperature_k = df["temperature_k"].to_numpy(dtype=float)
    temperature_err_k = _sanitize_nonnegative(
        df["temperature_err_total_k"].to_numpy(dtype=float)
    )
    carrier_density_cm3 = df["carrier_density_cm3"].to_numpy(dtype=float)
    if "carrier_density_err_total_cm3" in df.columns:
        carrier_density_err_cm3 = _sanitize_nonnegative(
            df["carrier_density_err_total_cm3"].to_numpy(dtype=float)
        )
    else:
        carrier_density_err_cm3 = np.zeros_like(carrier_density_cm3, dtype=float)

    e_laser_j, e_laser_ev = _laser_photon_energy_from_wavelength(laser_wavelength_nm)
    eg_j = eg_ev * E_CHARGE

    p_abs_w_cm2 = absorptivity_at_laser * p_exc_w_cm2
    p_abs_err_w_cm2 = np.abs(p_exc_w_cm2) * absorptivity_at_laser_sigma

    phi_abs_cm2_s = p_abs_w_cm2 / e_laser_j
    phi_abs_err_cm2_s = np.abs(p_exc_w_cm2 / e_laser_j) * absorptivity_at_laser_sigma
    phi_rad_cm2_s = plqy_eta * phi_abs_cm2_s
    phi_nonrad_cm2_s = (1.0 - plqy_eta) * phi_abs_cm2_s

    d_phi_rad_d_a = plqy_eta * p_exc_w_cm2 / e_laser_j
    d_phi_rad_d_eta = phi_abs_cm2_s
    phi_rad_err_cm2_s = np.sqrt(
        (d_phi_rad_d_a * absorptivity_at_laser_sigma) ** 2
        + (d_phi_rad_d_eta * plqy_eta_sigma) ** 2
    )

    d_phi_nonrad_d_a = (1.0 - plqy_eta) * p_exc_w_cm2 / e_laser_j
    d_phi_nonrad_d_eta = -phi_abs_cm2_s
    phi_nonrad_err_cm2_s = np.sqrt(
        (d_phi_nonrad_d_a * absorptivity_at_laser_sigma) ** 2
        + (d_phi_nonrad_d_eta * plqy_eta_sigma) ** 2
    )

    valid_temperature = np.isfinite(temperature_k) & (temperature_k > 0)
    beta_j = np.where(
        valid_temperature,
        eg_j + (3.0 - 2.0 * plqy_eta) * K_B * temperature_k,
        np.nan,
    )
    e_nonrad_j = np.where(valid_temperature, eg_j + 3.0 * K_B * temperature_k, np.nan)
    e_rad_j = np.where(valid_temperature, eg_j + K_B * temperature_k, np.nan)
    e_nonrad_ev = e_nonrad_j / E_CHARGE
    e_rad_ev = e_rad_j / E_CHARGE

    p_nonrad_w_cm2 = phi_nonrad_cm2_s * e_nonrad_j
    p_rad_w_cm2 = phi_rad_cm2_s * e_rad_j
    p_rec_w_cm2 = p_nonrad_w_cm2 + p_rad_w_cm2
    p_th_w_cm2 = p_abs_w_cm2 - p_rec_w_cm2

    prefactor = np.where(
        valid_temperature,
        absorptivity_at_laser * p_exc_w_cm2 / e_laser_j,
        np.nan,
    )
    cooling_factor = np.where(valid_temperature, 1.0 - beta_j / e_laser_j, np.nan)

    d_p_rec_d_a = np.where(valid_temperature, p_exc_w_cm2 * beta_j / e_laser_j, np.nan)
    d_p_rec_d_eta = np.where(valid_temperature, prefactor * (-2.0 * K_B * temperature_k), np.nan)
    d_p_rec_d_t = np.where(
        valid_temperature,
        prefactor * (3.0 - 2.0 * plqy_eta) * K_B,
        np.nan,
    )
    p_rec_err_w_cm2 = np.sqrt(
        (d_p_rec_d_a * absorptivity_at_laser_sigma) ** 2
        + (d_p_rec_d_eta * plqy_eta_sigma) ** 2
        + (d_p_rec_d_t * temperature_err_k) ** 2
    )

    d_p_th_d_a = np.where(valid_temperature, p_exc_w_cm2 * cooling_factor, np.nan)
    d_p_th_d_eta = np.where(valid_temperature, prefactor * (2.0 * K_B * temperature_k), np.nan)
    d_p_th_d_t = np.where(
        valid_temperature,
        -prefactor * (3.0 - 2.0 * plqy_eta) * K_B,
        np.nan,
    )
    p_th_err_w_cm2 = np.sqrt(
        (d_p_th_d_a * absorptivity_at_laser_sigma) ** 2
        + (d_p_th_d_eta * plqy_eta_sigma) ** 2
        + (d_p_th_d_t * temperature_err_k) ** 2
    )

    thickness_cm = active_layer_thickness_nm * 1e-7
    p_abs_w_cm3 = p_abs_w_cm2 / thickness_cm
    p_abs_err_w_cm3 = p_abs_err_w_cm2 / thickness_cm
    p_nonrad_w_cm3 = p_nonrad_w_cm2 / thickness_cm
    p_rad_w_cm3 = p_rad_w_cm2 / thickness_cm
    p_rec_w_cm3 = p_rec_w_cm2 / thickness_cm
    p_rec_err_w_cm3 = p_rec_err_w_cm2 / thickness_cm
    p_th_w_cm3 = p_th_w_cm2 / thickness_cm
    p_th_err_w_cm3 = p_th_err_w_cm2 / thickness_cm

    recombination_energy_per_pair_ev = beta_j / E_CHARGE
    thermalized_energy_per_pair_ev = np.where(
        valid_temperature,
        e_laser_ev - recombination_energy_per_pair_ev,
        np.nan,
    )

    thermalized_fraction = _safe_ratio(p_th_w_cm2, p_abs_w_cm2)
    recombination_fraction = _safe_ratio(p_rec_w_cm2, p_abs_w_cm2)
    radiative_fraction = _safe_ratio(p_rad_w_cm2, p_abs_w_cm2)
    nonradiative_fraction = _safe_ratio(p_nonrad_w_cm2, p_abs_w_cm2)
    thermalized_power_per_carrier_ev_s = _safe_ratio(
        p_th_w_cm3,
        carrier_density_cm3 * E_CHARGE,
    )
    rel_p_th = np.nan_to_num(_safe_ratio(p_th_err_w_cm3, np.abs(p_th_w_cm3)), nan=0.0)
    rel_n = np.nan_to_num(_safe_ratio(carrier_density_err_cm3, carrier_density_cm3), nan=0.0)
    thermalized_power_per_carrier_err_ev_s = (
        np.abs(thermalized_power_per_carrier_ev_s) * np.sqrt(rel_p_th**2 + rel_n**2)
    )

    _assign_dataframe_columns(
        df,
        {
            "laser_wavelength_nm": float(laser_wavelength_nm),
            "laser_photon_energy_ev": float(e_laser_ev),
            "absorptivity_at_laser": float(absorptivity_at_laser),
            "absorptivity_at_laser_sigma": float(absorptivity_at_laser_sigma),
            "plqy_eta": float(plqy_eta),
            "plqy_eta_sigma": float(plqy_eta_sigma),
            "active_layer_thickness_nm": float(active_layer_thickness_nm),
            "active_layer_thickness_cm": float(thickness_cm),
            "absorbed_power_w_cm2": p_abs_w_cm2,
            "absorbed_power_err_w_cm2": p_abs_err_w_cm2,
            "absorbed_power_w_cm3": p_abs_w_cm3,
            "absorbed_power_err_w_cm3": p_abs_err_w_cm3,
            "absorbed_photon_flux_cm2_s": phi_abs_cm2_s,
            "absorbed_photon_flux_err_cm2_s": phi_abs_err_cm2_s,
            "radiative_photon_flux_cm2_s": phi_rad_cm2_s,
            "radiative_photon_flux_err_cm2_s": phi_rad_err_cm2_s,
            "nonradiative_photon_flux_cm2_s": phi_nonrad_cm2_s,
            "nonradiative_photon_flux_err_cm2_s": phi_nonrad_err_cm2_s,
            "recombination_energy_nonrad_ev": e_nonrad_ev,
            "recombination_energy_rad_ev": e_rad_ev,
            "recombination_energy_avg_ev": recombination_energy_per_pair_ev,
            "thermalized_energy_per_pair_ev": thermalized_energy_per_pair_ev,
            "nonradiative_power_w_cm2": p_nonrad_w_cm2,
            "radiative_power_w_cm2": p_rad_w_cm2,
            "recombination_power_w_cm2": p_rec_w_cm2,
            "recombination_power_err_w_cm2": p_rec_err_w_cm2,
            "thermalized_power_w_cm2": p_th_w_cm2,
            "thermalized_power_err_w_cm2": p_th_err_w_cm2,
            "nonradiative_power_w_cm3": p_nonrad_w_cm3,
            "radiative_power_w_cm3": p_rad_w_cm3,
            "recombination_power_w_cm3": p_rec_w_cm3,
            "recombination_power_err_w_cm3": p_rec_err_w_cm3,
            "thermalized_power_w_cm3": p_th_w_cm3,
            "thermalized_power_err_w_cm3": p_th_err_w_cm3,
            "thermalized_power_per_carrier_ev_s": thermalized_power_per_carrier_ev_s,
            "thermalized_power_per_carrier_err_ev_s": thermalized_power_per_carrier_err_ev_s,
            "power_balance_closure_w_cm2": p_abs_w_cm2 - (p_th_w_cm2 + p_rec_w_cm2),
            "power_balance_closure_w_cm3": p_abs_w_cm3 - (p_th_w_cm3 + p_rec_w_cm3),
            "carrier_balance_closure_cm2_s": phi_abs_cm2_s - (phi_rad_cm2_s + phi_nonrad_cm2_s),
            "thermalized_fraction": thermalized_fraction,
            "recombination_fraction": recombination_fraction,
            "radiative_fraction": radiative_fraction,
            "nonradiative_fraction": nonradiative_fraction,
        },
    )
    return df


def auto_select_fit_window(
    energy_ev: np.ndarray,
    intensity: np.ndarray,
    assumed_a0: float,
) -> tuple[np.ndarray, float, float, str, np.ndarray]:
    min_points = max(3, WINDOW_MIN_POINTS)
    candidate_mask, window_mode = _build_scan_candidate_mask(
        energy_ev=energy_ev,
        intensity=intensity,
        min_points=min_points,
        prefer_peak_offset=True,
    )
    valid = np.isfinite(energy_ev) & np.isfinite(intensity) & (intensity > 0)
    samples = _enumerate_window_fit_samples(
        energy_ev=energy_ev,
        intensity=intensity,
        candidate_mask=candidate_mask,
        min_points=min_points,
        min_r2=WINDOW_MIN_R2,
        require_physical_bounds=True,
        assumed_a0=assumed_a0,
    )

    if samples:
        best_sample = min(
            samples,
            key=lambda sample: (sample.aicc, -sample.n_points, sample.fit_min_ev),
        )
        selected_idx = np.arange(best_sample.idx_start, best_sample.idx_end + 1)
        window_mode += "|aicc"
    else:
        selected_idx = np.flatnonzero(candidate_mask)
        if selected_idx.size == 0:
            selected_idx = np.flatnonzero(valid)
        window_mode += "|fallback_full_candidate"

    fit_mask = np.zeros_like(valid, dtype=bool)
    fit_mask[selected_idx] = True

    if selected_idx.size == 0:
        return fit_mask, np.nan, np.nan, window_mode, candidate_mask

    fit_min_ev = float(energy_ev[selected_idx[0]])
    fit_max_ev = float(energy_ev[selected_idx[-1]])
    return fit_mask, fit_min_ev, fit_max_ev, window_mode, candidate_mask


def _resolve_fit_window(
    energy_ev: np.ndarray,
    intensity: np.ndarray,
    auto_select_fit_window_enabled: bool,
    fit_min_ev_fixed: float,
    fit_max_ev_fixed: float,
    assumed_a0: float,
) -> tuple[np.ndarray, float, float, str, np.ndarray]:
    if auto_select_fit_window_enabled:
        return auto_select_fit_window(
            energy_ev=energy_ev,
            intensity=intensity,
            assumed_a0=assumed_a0,
        )

    valid = np.isfinite(energy_ev) & np.isfinite(intensity) & (intensity > 0)
    in_window = (energy_ev >= fit_min_ev_fixed) & (energy_ev <= fit_max_ev_fixed)
    fit_mask = valid & in_window
    scan_candidate_mask, _ = _build_scan_candidate_mask(
        energy_ev=energy_ev,
        intensity=intensity,
        min_points=max(3, FIT_RANGE_SCAN_MIN_POINTS),
        prefer_peak_offset=False,
    )
    return (
        fit_mask,
        fit_min_ev_fixed,
        fit_max_ev_fixed,
        "fixed",
        scan_candidate_mask,
    )


def fit_single_spectrum(
    energy_ev: np.ndarray,
    intensity: np.ndarray,
    spectrum_id: str,
    intensity_w_cm2: float,
    auto_select_fit_window_enabled: bool,
    fit_min_ev_fixed: float,
    fit_max_ev_fixed: float,
    assumed_a0: float,
    a0_sigma: float,
) -> tuple[
    FitResult,
    np.ndarray,
    list[tuple[float, float]],
    tuple[float, float] | None,
]:
    (
        fit_mask,
        fit_min_ev,
        fit_max_ev,
        window_mode,
        scan_candidate_mask,
    ) = _resolve_fit_window(
        energy_ev=energy_ev,
        intensity=intensity,
        auto_select_fit_window_enabled=auto_select_fit_window_enabled,
        fit_min_ev_fixed=fit_min_ev_fixed,
        fit_max_ev_fixed=fit_max_ev_fixed,
        assumed_a0=assumed_a0,
    )
    n_points_fit = int(np.count_nonzero(fit_mask))
    if n_points_fit < 3:
        empty_values = _empty_like_fit_result(
            spectrum_id=spectrum_id,
            intensity_w_cm2=intensity_w_cm2,
            a0_value=assumed_a0,
            a0_sigma=a0_sigma,
            fit_min_ev=fit_min_ev,
            fit_max_ev=fit_max_ev,
            window_mode=window_mode,
            n_points_fit=n_points_fit,
        )
        return FitResult(**empty_values), np.full_like(intensity, np.nan, dtype=float), [], None

    x_j = energy_ev[fit_mask] * E_CHARGE
    y = linearized_signal(energy_ev[fit_mask], intensity[fit_mask])
    slope, intercept, r2, covariance_line, _ = _compute_linear_fit_and_covariance(
        x_j, y
    )
    (
        temperature_k,
        qfls_effective_ev,
        qfls_ev,
        mu_e_ev,
        mu_h_ev,
        n_cm3,
    ) = _compute_parameters_from_line(
        slope=slope,
        intercept=intercept,
        assumed_a0=assumed_a0,
    )
    mu_e_fd_ev, mu_h_fd_ev, n_fd_cm3 = compute_mu_and_density_fd(
        temperature_k=temperature_k,
        qfls_ev=qfls_ev,
    )

    base_parameters = {
        "temperature_k": temperature_k,
        "qfls_effective_ev": qfls_effective_ev,
        "qfls_ev": qfls_ev,
        "mu_e_ev": mu_e_ev,
        "mu_h_ev": mu_h_ev,
        "carrier_density_cm3": n_cm3,
    }
    chi2_err = _chi2_parameter_uncertainties(
        slope=slope,
        intercept=intercept,
        covariance_line=covariance_line,
        temperature_k=temperature_k,
        qfls_ev=qfls_ev,
        assumed_a0=assumed_a0,
    )
    (
        range_err,
        n_windows_range,
        range_windows_ev,
        scan_domain_ev,
    ) = _fit_range_rms_uncertainty(
        energy_ev=energy_ev,
        intensity=intensity,
        base_fit_mask=fit_mask,
        base_parameters=base_parameters,
        assumed_a0=assumed_a0,
        scan_candidate_mask=scan_candidate_mask,
    )
    a0_err = _a0_parameter_uncertainties(
        temperature_k=temperature_k,
        qfls_ev=qfls_ev,
        assumed_a0=assumed_a0,
        a0_sigma=a0_sigma,
    )
    total_err = _combine_parameter_error_components(
        chi2_err=chi2_err,
        range_err=range_err,
        a0_err=a0_err,
    )
    intensity_model = _build_intensity_model(
        energy_ev=energy_ev,
        temperature_k=temperature_k,
        qfls_effective_ev=qfls_effective_ev,
    )

    result = _build_fit_result(
        spectrum_id=spectrum_id,
        intensity_w_cm2=intensity_w_cm2,
        a0_value=assumed_a0,
        a0_sigma=a0_sigma,
        fit_min_ev=fit_min_ev,
        fit_max_ev=fit_max_ev,
        window_mode=window_mode,
        n_points_fit=n_points_fit,
        slope=slope,
        intercept=intercept,
        r2=r2,
        base_parameters=base_parameters,
        mu_e_fd_ev=mu_e_fd_ev,
        mu_h_fd_ev=mu_h_fd_ev,
        carrier_density_fd_cm3=n_fd_cm3,
        chi2_err=chi2_err,
        range_err=range_err,
        a0_err=a0_err,
        total_err=total_err,
        fit_range_samples=n_windows_range,
    )
    return result, intensity_model, range_windows_ev, scan_domain_ev


def plot_raw_spectra(
    energy_ev: np.ndarray,
    spectra: np.ndarray,
    intensities_w_cm2: np.ndarray,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    norm = LogNorm(vmin=np.min(intensities_w_cm2), vmax=np.max(intensities_w_cm2))
    cmap = cm.cividis

    for i in range(spectra.shape[1]):
        ax.plot(
            energy_ev,
            spectra[:, i],
            color=cmap(norm(intensities_w_cm2[i])),
            lw=1.15,
            alpha=0.96,
        )

    style_axes(ax, logy=True)
    ax.set_xlabel(r"Photon energy, $E$ (eV)")
    ax.set_ylabel(r"PL intensity, $I_{\mathrm{PL}}$ (a.u.)")
    ax.set_title("GaAs photoluminescence spectra")
    ax.set_xlim(float(np.min(energy_ev)), float(np.max(energy_ev)))

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.018, fraction=0.045)
    cbar.set_label(r"Excitation intensity (W cm$^{-2}$)")
    cbar.ax.tick_params(direction="in")
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.08, top=0.95, hspace=0.2)
    save_figure(fig, outpath)
    plt.close(fig)


def plot_single_fit(
    energy_ev: np.ndarray,
    intensity: np.ndarray,
    intensity_model: np.ndarray,
    result: FitResult,
    fit_range_windows_ev: list[tuple[float, float]] | None,
    scan_domain_ev: tuple[float, float] | None,
    outpath: Path,
) -> None:
    fit_min_ev = result.fit_min_ev
    fit_max_ev = result.fit_max_ev
    fit_mask = (energy_ev >= fit_min_ev) & (energy_ev <= fit_max_ev) & (intensity > 0)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.3),
        sharex=False,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.16},
    )
    ax0, ax1 = axes

    scan_fill_color = "#ffe0b2"
    scan_edge_color = "#ef6c00"
    selected_fill_color = "#a5d6a7"
    selected_edge_color = "#2e7d32"
    envelope_fill_color = "#ce93d8"
    envelope_edge_color = "#6a1b9a"

    if scan_domain_ev is not None:
        ax0.axvspan(
            scan_domain_ev[0],
            scan_domain_ev[1],
            color=scan_fill_color,
            alpha=0.28,
            zorder=0,
            label="Full scan domain",
        )
        ax0.axvline(
            scan_domain_ev[0], color=scan_edge_color, lw=1.0, ls=":", alpha=0.9, zorder=3
        )
        ax0.axvline(
            scan_domain_ev[1], color=scan_edge_color, lw=1.0, ls=":", alpha=0.9, zorder=3
        )
    if fit_range_windows_ev:
        for lo_ev, hi_ev in fit_range_windows_ev:
            ax0.hlines(
                y=0.06,
                xmin=lo_ev,
                xmax=hi_ev,
                transform=ax0.get_xaxis_transform(),
                color=envelope_edge_color,
                lw=1.0,
                alpha=0.28,
                zorder=1,
            )
        lo_env = float(min(w[0] for w in fit_range_windows_ev))
        hi_env = float(max(w[1] for w in fit_range_windows_ev))
        coverage_pct = 100.0 * FIT_RANGE_SCAN_PLOT_WEIGHT_COVERAGE
        ax0.axvspan(
            lo_env,
            hi_env,
            facecolor=envelope_fill_color,
            alpha=0.16,
            hatch="///",
            edgecolor=envelope_edge_color,
            lw=0.9,
            zorder=1,
            label=f"{coverage_pct:.0f}% AICc-weight window envelope",
        )
        ax0.axvline(lo_env, color=envelope_edge_color, lw=0.9, ls="--", alpha=0.8, zorder=3)
        ax0.axvline(hi_env, color=envelope_edge_color, lw=0.9, ls="--", alpha=0.8, zorder=3)
    ax0.axvspan(
        fit_min_ev,
        fit_max_ev,
        facecolor=selected_fill_color,
        alpha=0.36,
        edgecolor=selected_edge_color,
        lw=0.95,
        zorder=2,
        label="Selected fit window",
    )
    ax0.axvline(fit_min_ev, color=selected_edge_color, lw=1.1, ls="-", alpha=0.95, zorder=4)
    ax0.axvline(fit_max_ev, color=selected_edge_color, lw=1.1, ls="-", alpha=0.95, zorder=4)
    ax0.plot(energy_ev, intensity, color="#1f4e79", lw=1.8, label="Experiment", zorder=5)
    ax0.plot(
        energy_ev,
        intensity_model,
        color="#d32f2f",
        lw=1.45,
        ls="--",
        label="High-energy GPL fit",
        zorder=6,
    )

    style_axes(ax0, logy=True)
    ax0.set_xlabel(r"Photon energy, $E$ (eV)")
    ax0.set_ylabel(r"PL intensity, $I_{\mathrm{PL}}$ (a.u.)")
    ax0.legend(loc="lower left", fontsize=9)
    ax0.set_title(
        f"Spectrum {result.spectrum_id}  |  "
        f"$I_{{exc}}$={result.intensity_w_cm2:.3g} W cm$^{{-2}}$"
    )
    t_label = f"{result.temperature_k:.1f}"
    if np.isfinite(result.temperature_err_total_k):
        t_label += rf"$\pm${result.temperature_err_total_k:.1f}"
    q_label = f"{result.qfls_ev:.3f}"
    if np.isfinite(result.qfls_err_total_ev):
        q_label += rf"$\pm${result.qfls_err_total_ev:.3f}"
    info_text = (
        r"$T$=" + f"{t_label} K, "
        + r"$\Delta\mu$=" + f"{q_label} eV, "
        + r"$R^2$=" + f"{result.r2:.5f}\n"
        + f"window=[{fit_min_ev:.3f}, {fit_max_ev:.3f}] eV"
        + f", scan={result.fit_range_samples:d}"
    )
    ax0.text(
        0.985,
        0.97,
        info_text,
        transform=ax0.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={
            "facecolor": "white",
            "edgecolor": "0.3",
            "boxstyle": "square,pad=0.25",
            "alpha": 0.95,
        },
    )

    y_all = linearized_signal(energy_ev[intensity > 0], intensity[intensity > 0])
    ax1.plot(energy_ev[intensity > 0], y_all, color="0.35", lw=1.05, label="Linearized data")

    x_fit_ev = energy_ev[fit_mask]
    x_fit_j = x_fit_ev * E_CHARGE
    y_line = result.slope * x_fit_j + result.intercept
    y_fit_data = linearized_signal(x_fit_ev, intensity[fit_mask])
    ax1.scatter(
        x_fit_ev,
        y_fit_data,
        s=13,
        color="#2e7d32",
        alpha=0.8,
        zorder=3,
        label="Points used for fit",
    )
    ax1.plot(x_fit_ev, y_line, color="#d32f2f", lw=1.5, ls="-", label="Linear regression")
    if scan_domain_ev is not None:
        ax1.axvspan(scan_domain_ev[0], scan_domain_ev[1], color="#b0bec5", alpha=0.12)
    ax1.axvspan(fit_min_ev, fit_max_ev, color="0.65", alpha=0.18)
    style_axes(ax1)
    ax1.set_xlabel(r"Photon energy, $E$ (eV)")
    ax1.set_ylabel(r"$\ln\!\left(\frac{h^3 c^2}{2E^2}I_{\mathrm{PL}}\right)$")
    ax1.legend(loc="best", fontsize=9)

    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.08, top=0.95, hspace=0.2)
    save_figure(fig, outpath)
    plt.close(fig)


def plot_summary(results_df: pd.DataFrame, outpath: Path) -> None:
    x = results_df["intensity_w_cm2"].to_numpy()
    x_valid = x[np.isfinite(x) & (x > 0)]
    if x_valid.size > 0:
        # Keep a small visual padding while using the full log-x width of the data range.
        x_pad_factor = 10 ** 0.02
        x_min_plot = float(np.min(x_valid) / x_pad_factor)
        x_max_plot = float(np.max(x_valid) * x_pad_factor)
    else:
        x_min_plot = np.nan
        x_max_plot = np.nan

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.8), sharex=True)
    ax00, ax01, ax10, ax11 = axes.ravel()

    ax00.errorbar(
        x,
        results_df["temperature_k"],
        yerr=results_df["temperature_err_total_k"],
        fmt="o-",
        lw=1.5,
        ms=4.5,
        capsize=2.5,
        elinewidth=1.0,
        color="#1565c0",
    )
    style_axes(ax00, logx=True)
    ax00.set_ylabel(r"Temperature, $T$ (K)")
    ax00.text(0.03, 0.93, "(a)", transform=ax00.transAxes, fontsize=11, fontweight="semibold")

    ax01.errorbar(
        x,
        results_df["qfls_ev"],
        yerr=results_df["qfls_err_total_ev"],
        fmt="o-",
        lw=1.5,
        ms=4.5,
        capsize=2.5,
        elinewidth=1.0,
        color="#6a1b9a",
        label=r"$\Delta\mu$",
    )
    ax01.errorbar(
        x,
        results_df["qfls_effective_ev"],
        yerr=results_df["qfls_effective_err_total_ev"],
        fmt="s--",
        lw=1.1,
        ms=3.5,
        capsize=2.0,
        elinewidth=0.9,
        color="#9c27b0",
        alpha=0.82,
        label=r"$\Delta\mu_{\mathrm{eff}}$",
    )
    style_axes(ax01, logx=True)
    ax01.set_ylabel(r"QFLS (eV)")
    ax01.legend(loc="best", fontsize=9)
    ax01.text(0.03, 0.93, "(b)", transform=ax01.transAxes, fontsize=11, fontweight="semibold")

    ax10.errorbar(
        x,
        results_df["mu_e_ev"],
        yerr=results_df["mu_e_err_total_ev"],
        fmt="o-",
        lw=1.5,
        ms=4.3,
        capsize=2.5,
        elinewidth=1.0,
        color="#ef6c00",
        label=r"$\mu_e$ (MB)",
    )
    ax10.errorbar(
        x,
        results_df["mu_h_ev"],
        yerr=results_df["mu_h_err_total_ev"],
        fmt="o-",
        lw=1.5,
        ms=4.3,
        capsize=2.5,
        elinewidth=1.0,
        color="#2e7d32",
        label=r"$\mu_h$ (MB)",
    )
    ax10.plot(
        x,
        results_df["mu_e_fd_ev"],
        "s--",
        lw=1.2,
        ms=3.9,
        color="#bf360c",
        alpha=0.9,
        label=r"$\mu_e$ (FD)",
    )
    ax10.plot(
        x,
        results_df["mu_h_fd_ev"],
        "s--",
        lw=1.2,
        ms=3.9,
        color="#1b5e20",
        alpha=0.9,
        label=r"$\mu_h$ (FD)",
    )
    style_axes(ax10, logx=True)
    ax10.set_xlabel(r"Excitation intensity, $I_{exc}$ (W cm$^{-2}$)")
    ax10.set_ylabel(r"Chemical potential (eV)")
    ax10.legend(loc="best", fontsize=9)
    ax10.text(0.03, 0.93, "(c)", transform=ax10.transAxes, fontsize=11, fontweight="semibold")

    n_vals = results_df["carrier_density_cm3"].to_numpy(dtype=float)
    n_err = results_df["carrier_density_err_total_cm3"].to_numpy(dtype=float)
    ax11.errorbar(
        x,
        n_vals,
        yerr=_safe_log_yerr(y=n_vals, err=n_err),
        fmt="o-",
        lw=1.5,
        ms=4.5,
        capsize=2.5,
        elinewidth=1.0,
        color="#00838f",
        label=r"$n$ (MB)",
    )
    n_fd_vals = results_df["carrier_density_fd_cm3"].to_numpy(dtype=float)
    ax11.plot(
        x,
        n_fd_vals,
        "s--",
        lw=1.2,
        ms=4.0,
        color="#004d40",
        alpha=0.9,
        label=r"$n$ (FD)",
    )
    style_axes(ax11, logx=True, logy=True)
    ax11.set_xlabel(r"Excitation intensity, $I_{exc}$ (W cm$^{-2}$)")
    ax11.set_ylabel(r"Carrier density, $n$ (cm$^{-3}$)")
    ax11.legend(loc="best", fontsize=9)
    ax11.text(0.03, 0.93, "(d)", transform=ax11.transAxes, fontsize=11, fontweight="semibold")

    if np.isfinite(x_min_plot) and np.isfinite(x_max_plot) and (x_max_plot > x_min_plot):
        for ax in (ax00, ax01, ax10, ax11):
            ax.set_xlim(x_min_plot, x_max_plot)

    fig.suptitle(
        "Extracted hot-carrier parameters versus excitation intensity",
        y=1.01,
        fontsize=13,
    )
    fig.tight_layout(pad=0.7)
    save_figure(fig, outpath)
    plt.close(fig)


def load_tsai_model_table(table_csv: str) -> pd.DataFrame | None:
    if not table_csv:
        return None

    table_path = Path(table_csv).expanduser()
    if not table_path.is_file():
        print(f"Warning: Tsai model table not found, skipping: {table_path}")
        return None

    model_df = pd.read_csv(table_path)
    required_columns = {"n_cm3", "temperature_k", "p_th_w_cm3"}
    missing = sorted(required_columns - set(model_df.columns))
    if missing:
        raise ValueError(
            "Tsai model table is missing required columns: "
            + ", ".join(missing)
        )

    model_df = model_df[list(required_columns)].copy()
    model_df = model_df.apply(pd.to_numeric, errors="coerce")
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
    model_df = model_df[
        (model_df["n_cm3"] > 0)
        & (model_df["temperature_k"] > 0)
        & (model_df["p_th_w_cm3"] > 0)
    ]
    if model_df.shape[0] < 3:
        print("Warning: Tsai model table has fewer than 3 valid points; skipping.")
        return None
    return model_df


def _nearest_theory_prediction(
    exp_n_cm3: np.ndarray,
    exp_t_k: np.ndarray,
    theory_n_cm3: np.ndarray,
    theory_t_k: np.ndarray,
    theory_pth_w_cm3: np.ndarray,
) -> np.ndarray:
    if theory_n_cm3.size == 0:
        return np.full_like(exp_n_cm3, np.nan, dtype=float)

    exp_log_n = np.log10(exp_n_cm3)[:, None]
    theory_log_n = np.log10(theory_n_cm3)[None, :]
    exp_t = exp_t_k[:, None]
    theory_t = theory_t_k[None, :]

    scale_log_n = max(float(np.ptp(np.log10(theory_n_cm3))), 1e-9)
    scale_t = max(float(np.ptp(theory_t_k)), 1e-9)
    dist2 = ((exp_log_n - theory_log_n) / scale_log_n) ** 2 + (
        (exp_t - theory_t) / scale_t
    ) ** 2
    nearest_idx = np.argmin(dist2, axis=1)
    return theory_pth_w_cm3[nearest_idx]


def plot_pth_nt_comparison(
    results_df: pd.DataFrame,
    outpath: Path,
    theory_df: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    n_cm3 = results_df["carrier_density_cm3"].to_numpy(dtype=float)
    n_err_cm3 = results_df["carrier_density_err_total_cm3"].to_numpy(dtype=float)
    temperature_k = results_df["temperature_k"].to_numpy(dtype=float)
    temperature_err_k = results_df["temperature_err_total_k"].to_numpy(dtype=float)
    p_th_w_cm3 = results_df["thermalized_power_w_cm3"].to_numpy(dtype=float)
    p_th_err_w_cm3 = results_df["thermalized_power_err_w_cm3"].to_numpy(dtype=float)
    intensity = results_df["intensity_w_cm2"].to_numpy(dtype=float)

    valid = (
        np.isfinite(n_cm3)
        & np.isfinite(temperature_k)
        & np.isfinite(p_th_w_cm3)
        & (n_cm3 > 0)
        & (temperature_k > 0)
        & (p_th_w_cm3 > 0)
    )
    if np.count_nonzero(valid) < 3:
        return None

    n_plot = n_cm3[valid]
    n_err_plot = np.where(
        np.isfinite(n_err_cm3[valid]) & (n_err_cm3[valid] >= 0),
        n_err_cm3[valid],
        0.0,
    )
    t_plot = temperature_k[valid]
    t_err_plot = np.where(
        np.isfinite(temperature_err_k[valid]) & (temperature_err_k[valid] >= 0),
        temperature_err_k[valid],
        0.0,
    )
    p_th_plot = p_th_w_cm3[valid]
    p_th_err_plot = np.where(
        np.isfinite(p_th_err_w_cm3[valid]) & (p_th_err_w_cm3[valid] >= 0),
        p_th_err_w_cm3[valid],
        0.0,
    )
    intensity_plot = intensity[valid]

    pth_norm = LogNorm(vmin=float(np.min(p_th_plot)), vmax=float(np.max(p_th_plot)))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.8))

    ax0.errorbar(
        n_plot,
        t_plot,
        xerr=n_err_plot,
        yerr=t_err_plot,
        fmt="none",
        ecolor="0.6",
        alpha=0.38,
        elinewidth=0.7,
        capsize=1.8,
        zorder=1,
    )
    s0 = ax0.scatter(
        n_plot,
        t_plot,
        c=p_th_plot,
        cmap="viridis",
        norm=pth_norm,
        s=55,
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
    )

    if theory_df is not None:
        theory_n = theory_df["n_cm3"].to_numpy(dtype=float)
        theory_t = theory_df["temperature_k"].to_numpy(dtype=float)
        theory_p = theory_df["p_th_w_cm3"].to_numpy(dtype=float)
        try:
            tri = Triangulation(theory_n, theory_t)
            level_min = max(float(np.min(theory_p)), float(np.min(p_th_plot)))
            level_max = min(float(np.max(theory_p)), float(np.max(p_th_plot)))
            if level_max > level_min:
                levels = np.geomspace(level_min, level_max, 7)
                contour = ax0.tricontour(
                    tri,
                    theory_p,
                    levels=levels,
                    colors="white",
                    linewidths=1.0,
                    alpha=0.9,
                )
                ax0.clabel(contour, inline=True, fmt="%.2e", fontsize=7)
        except RuntimeError:
            ax0.text(
                0.03,
                0.04,
                "Tsai contour overlay skipped\n(non-triangulable model grid)",
                transform=ax0.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.5,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "0.45",
                    "boxstyle": "square,pad=0.2",
                    "alpha": 0.85,
                },
            )

    style_axes(ax0, logx=True)
    ax0.set_xlabel(r"Carrier density, $n$ (cm$^{-3}$)")
    ax0.set_ylabel(r"Carrier temperature, $T$ (K)")
    ax0.set_title(r"Experimental manifold in $(n,T)$ colored by $P_{\mathrm{th}}$")
    cbar0 = fig.colorbar(s0, ax=ax0, pad=0.02, fraction=0.05)
    cbar0.set_label(r"$P_{\mathrm{th}}$ (W cm$^{-3}$)")

    comparison_df: pd.DataFrame | None = None
    if theory_df is not None:
        theory_n = theory_df["n_cm3"].to_numpy(dtype=float)
        theory_t = theory_df["temperature_k"].to_numpy(dtype=float)
        theory_p = theory_df["p_th_w_cm3"].to_numpy(dtype=float)
        p_th_theory_at_exp = _nearest_theory_prediction(
            exp_n_cm3=n_plot,
            exp_t_k=t_plot,
            theory_n_cm3=theory_n,
            theory_t_k=theory_t,
            theory_pth_w_cm3=theory_p,
        )
        valid_cmp = (
            np.isfinite(p_th_theory_at_exp)
            & (p_th_theory_at_exp > 0)
            & np.isfinite(p_th_plot)
            & (p_th_plot > 0)
        )
        if np.count_nonzero(valid_cmp) >= 2:
            ax1.errorbar(
                p_th_plot[valid_cmp],
                p_th_theory_at_exp[valid_cmp],
                xerr=_safe_log_yerr(
                    y=p_th_plot[valid_cmp],
                    err=p_th_err_plot[valid_cmp],
                ),
                fmt="none",
                ecolor="0.55",
                alpha=0.35,
                elinewidth=0.8,
                capsize=1.8,
            )
            s1 = ax1.scatter(
                p_th_plot[valid_cmp],
                p_th_theory_at_exp[valid_cmp],
                c=intensity_plot[valid_cmp],
                cmap="cividis",
                s=52,
                edgecolors="white",
                linewidths=0.5,
            )
            xy_min = float(
                min(
                    np.min(p_th_plot[valid_cmp]),
                    np.min(p_th_theory_at_exp[valid_cmp]),
                )
            )
            xy_max = float(
                max(
                    np.max(p_th_plot[valid_cmp]),
                    np.max(p_th_theory_at_exp[valid_cmp]),
                )
            )
            line = np.geomspace(xy_min * 0.9, xy_max * 1.1, 160)
            ax1.plot(line, line, "--", color="0.25", lw=1.1, label="1:1 line")
            style_axes(ax1, logx=True, logy=True)
            ax1.set_xlabel(r"Experimental $P_{\mathrm{th}}$ (W cm$^{-3}$)")
            ax1.set_ylabel(r"Tsai-model $P_{\mathrm{th}}$ (W cm$^{-3}$)")
            ax1.set_title("Direct pointwise comparison at measured $(n,T)$")
            ax1.legend(loc="best", fontsize=8.5)
            cbar1 = fig.colorbar(s1, ax=ax1, pad=0.02, fraction=0.05)
            cbar1.set_label(r"$I_{\mathrm{exc}}$ (W cm$^{-2}$)")
            comparison_df = pd.DataFrame(
                {
                    "carrier_density_cm3": n_plot[valid_cmp],
                    "temperature_k": t_plot[valid_cmp],
                    "intensity_w_cm2": intensity_plot[valid_cmp],
                    "pth_experiment_w_cm3": p_th_plot[valid_cmp],
                    "pth_experiment_err_w_cm3": p_th_err_plot[valid_cmp],
                    "pth_tsai_nearest_w_cm3": p_th_theory_at_exp[valid_cmp],
                    "pth_ratio_tsai_over_exp": (
                        p_th_theory_at_exp[valid_cmp] / p_th_plot[valid_cmp]
                    ),
                }
            )
        else:
            style_axes(ax1)
            ax1.set_title("Direct pointwise comparison at measured $(n,T)$")
            ax1.text(
                0.5,
                0.5,
                "Insufficient valid overlap\nbetween experiment and Tsai table",
                transform=ax1.transAxes,
                ha="center",
                va="center",
                fontsize=9,
            )
            ax1.set_xticks([])
            ax1.set_yticks([])
    else:
        s1 = ax1.scatter(
            n_plot,
            p_th_plot,
            c=t_plot,
            cmap="plasma",
            s=56,
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
        )
        ax1.errorbar(
            n_plot,
            p_th_plot,
            xerr=n_err_plot,
            yerr=_safe_log_yerr(y=p_th_plot, err=p_th_err_plot),
            fmt="none",
            ecolor="0.55",
            alpha=0.35,
            elinewidth=0.8,
            capsize=1.8,
            zorder=1,
        )
        style_axes(ax1, logx=True, logy=True)
        ax1.set_xlabel(r"Carrier density, $n$ (cm$^{-3}$)")
        ax1.set_ylabel(r"$P_{\mathrm{th}}$ (W cm$^{-3}$)")
        ax1.set_title(r"Experimental $P_{\mathrm{th}}(n)$, color-coded by $T$")
        cbar1 = fig.colorbar(s1, ax=ax1, pad=0.02, fraction=0.05)
        cbar1.set_label("Temperature (K)")

    fig.suptitle(r"Comparison-ready representation of $P_{\mathrm{th}}(n,T)$", y=1.02)
    fig.tight_layout(pad=0.7)
    save_figure(fig, outpath)
    plt.close(fig)
    return comparison_df


def plot_thermalized_power_diagnostics(results_df: pd.DataFrame, outpath: Path) -> None:
    n_cm3 = results_df["carrier_density_cm3"].to_numpy(dtype=float)
    n_err_cm3 = _sanitize_nonnegative(
        results_df.get(
            "carrier_density_err_total_cm3",
            pd.Series(np.zeros(results_df.shape[0], dtype=float)),
        ).to_numpy(dtype=float)
    )
    temperature_k = results_df["temperature_k"].to_numpy(dtype=float)
    temperature_err_k = _sanitize_nonnegative(
        results_df.get(
            "temperature_err_total_k",
            pd.Series(np.zeros(results_df.shape[0], dtype=float)),
        ).to_numpy(dtype=float)
    )
    p_th_w_cm3 = results_df["thermalized_power_w_cm3"].to_numpy(dtype=float)
    p_th_err_w_cm3 = _sanitize_nonnegative(
        results_df.get(
            "thermalized_power_err_w_cm3",
            pd.Series(np.zeros(results_df.shape[0], dtype=float)),
        ).to_numpy(dtype=float)
    )
    p_th_per_carrier_ev_s = results_df["thermalized_power_per_carrier_ev_s"].to_numpy(dtype=float)
    p_th_per_carrier_err_ev_s = _sanitize_nonnegative(
        results_df.get(
            "thermalized_power_per_carrier_err_ev_s",
            pd.Series(np.zeros(results_df.shape[0], dtype=float)),
        ).to_numpy(dtype=float)
    )
    thermalized_energy_pair_ev = results_df["thermalized_energy_per_pair_ev"].to_numpy(dtype=float)
    intensity_w_cm2 = results_df["intensity_w_cm2"].to_numpy(dtype=float)

    valid = (
        np.isfinite(n_cm3)
        & np.isfinite(temperature_k)
        & np.isfinite(p_th_w_cm3)
        & np.isfinite(p_th_per_carrier_ev_s)
        & np.isfinite(thermalized_energy_pair_ev)
        & (n_cm3 > 0)
        & (temperature_k > 0)
        & (p_th_w_cm3 > 0)
        & (p_th_per_carrier_ev_s > 0)
    )
    if np.count_nonzero(valid) < 3:
        return

    n_plot = n_cm3[valid]
    n_err_plot = n_err_cm3[valid]
    t_plot = temperature_k[valid]
    t_err_plot = temperature_err_k[valid]
    p_th_plot = p_th_w_cm3[valid]
    p_th_err_plot = p_th_err_w_cm3[valid]
    p_th_per_carrier_plot = p_th_per_carrier_ev_s[valid]
    p_th_per_carrier_err_plot = p_th_per_carrier_err_ev_s[valid]
    thermalized_energy_plot = thermalized_energy_pair_ev[valid]
    intensity_plot = intensity_w_cm2[valid]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.3))
    ax00, ax01 = axes[0]
    ax10, ax11 = axes[1]

    ax00.errorbar(
        n_plot,
        p_th_plot,
        xerr=n_err_plot,
        yerr=_safe_log_yerr(y=p_th_plot, err=p_th_err_plot),
        fmt="none",
        ecolor="0.6",
        alpha=0.35,
        elinewidth=0.8,
        capsize=1.8,
        zorder=1,
    )
    s00 = ax00.scatter(
        n_plot,
        p_th_plot,
        c=t_plot,
        cmap="viridis",
        s=56,
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
    )
    style_axes(ax00, logx=True, logy=True)
    ax00.set_xlabel(r"Carrier density, $n$ (cm$^{-3}$)")
    ax00.set_ylabel(r"$P_{\mathrm{th}}$ (W cm$^{-3}$)")
    ax00.set_title(r"Volumetric thermalized power across carrier states")
    cbar00 = fig.colorbar(s00, ax=ax00, pad=0.02, fraction=0.052)
    cbar00.set_label("Temperature (K)")
    ax00.text(0.03, 0.93, "(a)", transform=ax00.transAxes, fontsize=11, fontweight="semibold")

    norm_n = LogNorm(vmin=float(np.min(n_plot)), vmax=float(np.max(n_plot)))
    ax01.errorbar(
        t_plot,
        p_th_plot,
        xerr=t_err_plot,
        yerr=_safe_log_yerr(y=p_th_plot, err=p_th_err_plot),
        fmt="none",
        ecolor="0.6",
        alpha=0.35,
        elinewidth=0.8,
        capsize=1.8,
        zorder=1,
    )
    s01 = ax01.scatter(
        t_plot,
        p_th_plot,
        c=n_plot,
        cmap="cividis",
        norm=norm_n,
        s=56,
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
    )
    style_axes(ax01, logy=True)
    ax01.set_xlabel("Carrier temperature, $T$ (K)")
    ax01.set_ylabel(r"$P_{\mathrm{th}}$ (W cm$^{-3}$)")
    ax01.set_title(r"Thermalized power versus carrier temperature")
    cbar01 = fig.colorbar(s01, ax=ax01, pad=0.02, fraction=0.052)
    cbar01.set_label(r"$n$ (cm$^{-3}$)")
    ax01.text(0.03, 0.93, "(b)", transform=ax01.transAxes, fontsize=11, fontweight="semibold")

    ax10.errorbar(
        n_plot,
        p_th_per_carrier_plot,
        xerr=n_err_plot,
        yerr=_safe_log_yerr(y=p_th_per_carrier_plot, err=p_th_per_carrier_err_plot),
        fmt="none",
        ecolor="0.6",
        alpha=0.35,
        elinewidth=0.8,
        capsize=1.8,
        zorder=1,
    )
    s10 = ax10.scatter(
        n_plot,
        p_th_per_carrier_plot,
        c=t_plot,
        cmap="plasma",
        s=56,
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
    )
    style_axes(ax10, logx=True, logy=True)
    ax10.set_xlabel(r"Carrier density, $n$ (cm$^{-3}$)")
    ax10.set_ylabel(r"$P_{\mathrm{th}}/n$ (eV s$^{-1}$ carrier$^{-1}$)")
    ax10.set_title(r"Per-carrier cooling rate versus carrier density")
    cbar10 = fig.colorbar(s10, ax=ax10, pad=0.02, fraction=0.052)
    cbar10.set_label("Temperature (K)")
    ax10.text(0.03, 0.93, "(c)", transform=ax10.transAxes, fontsize=11, fontweight="semibold")

    intensity_norm = LogNorm(vmin=float(np.min(intensity_plot)), vmax=float(np.max(intensity_plot)))
    ax11.errorbar(
        t_plot,
        thermalized_energy_plot,
        xerr=t_err_plot,
        fmt="none",
        ecolor="0.6",
        alpha=0.35,
        elinewidth=0.8,
        capsize=1.8,
        zorder=1,
    )
    s11 = ax11.scatter(
        t_plot,
        thermalized_energy_plot,
        c=intensity_plot,
        cmap="magma",
        norm=intensity_norm,
        s=56,
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
    )
    e_laser_ev = float(results_df["laser_photon_energy_ev"].to_numpy(dtype=float)[0])
    eta = float(results_df["plqy_eta"].to_numpy(dtype=float)[0])
    t_line = np.linspace(float(np.min(t_plot)) * 0.98, float(np.max(t_plot)) * 1.02, 160)
    delta_e_line = e_laser_ev - (
        EG_EV + (3.0 - 2.0 * eta) * (K_B / E_CHARGE) * t_line
    )
    ax11.plot(
        t_line,
        delta_e_line,
        "--",
        lw=1.2,
        color="0.2",
        label=r"$E_{laser}-(E_g+(3-2\eta)k_BT)$",
    )
    style_axes(ax11)
    ax11.set_xlabel("Carrier temperature, $T$ (K)")
    ax11.set_ylabel(r"Thermalized energy per pair (eV)")
    ax11.set_title(r"Excess energy dissipated per absorbed carrier pair")
    ax11.legend(loc="best", fontsize=8.5)
    cbar11 = fig.colorbar(s11, ax=ax11, pad=0.02, fraction=0.052)
    cbar11.set_label(r"$I_{\mathrm{exc}}$ (W cm$^{-2}$)")
    ax11.text(0.03, 0.93, "(d)", transform=ax11.transAxes, fontsize=11, fontweight="semibold")

    fig.suptitle(r"Thermalized-power diagnostics in carrier-state space", y=1.01)
    fig.tight_layout(pad=0.8)
    save_figure(fig, outpath)
    plt.close(fig)


def _validate_configuration() -> None:
    checks = (
        (ASSUMED_A0 > 0, "ASSUMED_A0 must be strictly positive."),
        (A0_SIGMA >= 0, "A0_SIGMA must be non-negative."),
        (LASER_WAVELENGTH_NM > 0, "LASER_WAVELENGTH_NM must be strictly positive."),
        (
            0 <= ABSORPTIVITY_AT_LASER <= 1,
            "ABSORPTIVITY_AT_LASER must be in [0, 1].",
        ),
        (
            ABSORPTIVITY_AT_LASER_SIGMA >= 0,
            "ABSORPTIVITY_AT_LASER_SIGMA must be non-negative.",
        ),
        (0 <= PLQY_ETA <= 1, "PLQY_ETA must be in [0, 1]."),
        (PLQY_ETA_SIGMA >= 0, "PLQY_ETA_SIGMA must be non-negative."),
        (
            ACTIVE_LAYER_THICKNESS_NM > 0,
            "ACTIVE_LAYER_THICKNESS_NM must be strictly positive.",
        ),
        (
            0 < FIT_RANGE_SCAN_PLOT_WEIGHT_COVERAGE <= 1,
            "FIT_RANGE_SCAN_PLOT_WEIGHT_COVERAGE must be in the interval (0, 1].",
        ),
    )
    for is_valid, message in checks:
        if not is_valid:
            raise ValueError(message)


def _fit_all_spectra(
    energy_ev: np.ndarray,
    spectra: np.ndarray,
    spectrum_ids: list[str],
    intensities_w_cm2: np.ndarray,
    fit_dir: Path,
) -> list[FitResult]:
    all_results: list[FitResult] = []
    for i, (spectrum_id, intensity_w_cm2) in enumerate(
        zip(spectrum_ids, intensities_w_cm2, strict=True)
    ):
        result, intensity_model, range_windows, scan_domain = fit_single_spectrum(
            energy_ev=energy_ev,
            intensity=spectra[:, i],
            spectrum_id=spectrum_id,
            intensity_w_cm2=float(intensity_w_cm2),
            auto_select_fit_window_enabled=AUTO_SELECT_FIT_WINDOW,
            fit_min_ev_fixed=FIT_ENERGY_MIN_EV,
            fit_max_ev_fixed=FIT_ENERGY_MAX_EV,
            assumed_a0=ASSUMED_A0,
            a0_sigma=A0_SIGMA,
        )
        all_results.append(result)
        plot_single_fit(
            energy_ev=energy_ev,
            intensity=spectra[:, i],
            intensity_model=intensity_model,
            result=result,
            fit_range_windows_ev=range_windows,
            scan_domain_ev=scan_domain,
            outpath=fit_dir / f"fit_spectrum_{i:02d}.png",
        )
    return all_results


def _print_run_summary(
    out_dir: Path,
    fit_dir: Path,
    comparison_df: pd.DataFrame | None,
    tsai_model_df: pd.DataFrame | None,
) -> None:
    print("Done.")
    print(f"Raw spectra plot: {out_dir / 'all_spectra_logscale.png'}")
    print(f"Spectrum fits:    {fit_dir}")
    print(f"Results table:    {out_dir / 'fit_results.csv'}")
    print(f"Summary figure:   {out_dir / 'parameters_vs_intensity.png'}")
    print(f"Power figure:     {out_dir / 'thermalized_power_diagnostics.png'}")
    print(f"Pth(n,T) figure:  {out_dir / 'pth_nT_comparison.png'}")
    if comparison_df is not None:
        print(f"Tsai compare CSV: {out_dir / 'pth_experiment_vs_tsai.csv'}")
    if AUTO_SELECT_FIT_WINDOW:
        print(
            "Fit window mode:  AUTO per spectrum | "
            f"search=[{WINDOW_SEARCH_MIN_EV:.3f}, {WINDOW_SEARCH_MAX_EV:.3f}] eV, "
            f"peak_offset={WINDOW_PEAK_OFFSET_EV:.3f} eV"
        )
    else:
        print(
            f"Fit window used:  [{FIT_ENERGY_MIN_EV:.3f}, {FIT_ENERGY_MAX_EV:.3f}] eV | "
            f"ASSUMED_A0={ASSUMED_A0:g}"
        )
    if ESTIMATE_FIT_RANGE_UNCERTAINTY:
        print(
            "Uncertainty mode: chi^2 + AICc-weighted fit-range + A0 | "
            f"min_points={FIT_RANGE_SCAN_MIN_POINTS}, "
            f"min_r2={FIT_RANGE_SCAN_MIN_R2:.3f}, "
            f"plot_weight_coverage={FIT_RANGE_SCAN_PLOT_WEIGHT_COVERAGE:.2f}, "
            f"A0={ASSUMED_A0:.4f}±{A0_SIGMA:.4f} ({A0_UNCERTAINTY_MODEL})"
        )
    else:
        print(
            "Uncertainty mode: chi^2 + A0 | "
            f"A0={ASSUMED_A0:.4f}±{A0_SIGMA:.4f} ({A0_UNCERTAINTY_MODEL})"
        )
    print(
        "Power model:      "
        rf"lambda_laser={LASER_WAVELENGTH_NM:.1f} nm, "
        rf"A_laser={ABSORPTIVITY_AT_LASER:.4f}±{ABSORPTIVITY_AT_LASER_SIGMA:.4f}, "
        rf"PLQY eta={PLQY_ETA:.4f}±{PLQY_ETA_SIGMA:.4f}, "
        rf"d={ACTIVE_LAYER_THICKNESS_NM:.1f} nm"
    )
    if tsai_model_df is None:
        print("Tsai model table: not provided (set TSAI_MODEL_TABLE_CSV to enable overlay/parity)")
    else:
        print(
            "Tsai model table: "
            f"{TSAI_MODEL_TABLE_CSV} | points={tsai_model_df.shape[0]}"
        )


def main() -> None:
    setup_plot_style()
    _validate_configuration()

    root = Path(__file__).resolve().parent
    out_dir = root / "outputs"
    fit_dir = out_dir / "fits"
    out_dir.mkdir(exist_ok=True)
    fit_dir.mkdir(exist_ok=True)

    df_pl = load_spectra(root, FILENAME)
    energy_ev = df_pl.index.to_numpy(dtype=float)
    spectra = df_pl.to_numpy(dtype=float)
    spectrum_ids = list(df_pl.columns.astype(str))

    if len(EXCITATION_INTENSITY_W_CM2) != spectra.shape[1]:
        raise ValueError(
            "Intensity list has "
            f"{len(EXCITATION_INTENSITY_W_CM2)} values but file has "
            f"{spectra.shape[1]} spectra."
        )

    # Sort by increasing energy for cleaner plots and fits
    sort_idx = np.argsort(energy_ev)
    energy_ev = energy_ev[sort_idx]
    spectra = spectra[sort_idx, :]

    plot_raw_spectra(
        energy_ev=energy_ev,
        spectra=spectra,
        intensities_w_cm2=EXCITATION_INTENSITY_W_CM2,
        outpath=out_dir / "all_spectra_logscale.png",
    )

    all_results = _fit_all_spectra(
        energy_ev=energy_ev,
        spectra=spectra,
        spectrum_ids=spectrum_ids,
        intensities_w_cm2=EXCITATION_INTENSITY_W_CM2,
        fit_dir=fit_dir,
    )

    results_df = pd.DataFrame([r.__dict__ for r in all_results])
    results_df = compute_power_balance_table(
        results_df=results_df,
        laser_wavelength_nm=LASER_WAVELENGTH_NM,
        absorptivity_at_laser=ABSORPTIVITY_AT_LASER,
        absorptivity_at_laser_sigma=ABSORPTIVITY_AT_LASER_SIGMA,
        plqy_eta=PLQY_ETA,
        plqy_eta_sigma=PLQY_ETA_SIGMA,
        active_layer_thickness_nm=ACTIVE_LAYER_THICKNESS_NM,
        eg_ev=EG_EV,
    )
    tsai_model_df = load_tsai_model_table(TSAI_MODEL_TABLE_CSV)
    comparison_df = plot_pth_nt_comparison(
        results_df=results_df,
        outpath=out_dir / "pth_nT_comparison.png",
        theory_df=tsai_model_df,
    )
    results_df.to_csv(out_dir / "fit_results.csv", index=False)
    plot_summary(results_df, out_dir / "parameters_vs_intensity.png")
    legacy_power_plot = out_dir / "thermalized_power_vs_absorbed.png"
    if legacy_power_plot.exists():
        legacy_power_plot.unlink()
    plot_thermalized_power_diagnostics(results_df, out_dir / "thermalized_power_diagnostics.png")
    if comparison_df is not None:
        comparison_df.to_csv(out_dir / "pth_experiment_vs_tsai.csv", index=False)
    _print_run_summary(
        out_dir=out_dir,
        fit_dir=fit_dir,
        comparison_df=comparison_df,
        tsai_model_df=tsai_model_df,
    )


if __name__ == "__main__":
    main()
