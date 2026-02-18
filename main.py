from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter


# ----------------------------- User-tunable settings -----------------------------
FILENAME = "GaAs bulk_PL_avg_circle_4pixs.txt"

# If False, fixed fit window below is used for all spectra.
AUTO_SELECT_FIT_WINDOW = True
FIT_ENERGY_MIN_EV = 1.55
FIT_ENERGY_MAX_EV = 1.70

# Auto-window selection parameters
WINDOW_SEARCH_MIN_EV = 1.55
WINDOW_SEARCH_MAX_EV = 1.70
WINDOW_PEAK_OFFSET_EV = 0.045
WINDOW_MIN_POINTS = 18
WINDOW_MIN_R2 = 0.995
WINDOW_T_MIN_K = 150.0
WINDOW_T_MAX_K = 1200.0
WINDOW_LENGTH_WEIGHT = 2.0e-4
WINDOW_HIGH_ENERGY_WEIGHT = 2.0e-4

# Fit-range uncertainty scan around selected fit bounds (in index points)
ESTIMATE_FIT_RANGE_UNCERTAINTY = True
FIT_RANGE_SCAN_SHIFT_POINTS = 4
FIT_RANGE_SCAN_MIN_POINTS = 12
FIT_RANGE_SCAN_MIN_R2 = 0.99
FIT_RANGE_SCAN_REQUIRE_PHYSICAL_BOUNDS = True

# Figure export/style
SAVE_DPI = 450
EXPORT_PDF = True

# A0 is usually unknown in the simplified linear fit.
# If left at 1.0, reported QFLS is an effective value (offset absorbed in A0).
ASSUMED_A0 = 1.0

# GaAs parameters for optional Maxwell-Boltzmann carrier estimates
EG_EV = 1.424
M_E_EFF = 0.067
M_H_EFF = 0.50

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
    temperature_err_chi2_k: float
    temperature_err_range_k: float
    temperature_err_total_k: float
    qfls_effective_err_chi2_ev: float
    qfls_effective_err_range_ev: float
    qfls_effective_err_total_ev: float
    qfls_err_chi2_ev: float
    qfls_err_range_ev: float
    qfls_err_total_ev: float
    mu_e_err_chi2_ev: float
    mu_e_err_range_ev: float
    mu_e_err_total_ev: float
    mu_h_err_chi2_ev: float
    mu_h_err_range_ev: float
    mu_h_err_total_ev: float
    carrier_density_err_chi2_cm3: float
    carrier_density_err_range_cm3: float
    carrier_density_err_total_cm3: float
    fit_range_samples: int


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
    fig.savefig(outpath, dpi=SAVE_DPI, bbox_inches="tight")
    if EXPORT_PDF:
        fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")


def load_spectra(data_dir: Path, filename: str) -> pd.DataFrame:
    foldSave = str(data_dir.resolve()) + "\\"
    dfPL = pd.read_csv(foldSave + filename, sep=";", index_col=0)
    return dfPL


def linearized_signal(energy_ev: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    energy_j = energy_ev * E_CHARGE
    return np.log((H**3 * C**2 / (2.0 * energy_j**2)) * intensity)


def compute_mu_and_density_mb(
    temperature_k: float,
    qfls_ev: float,
    eg_ev: float = EG_EV,
    m_e_eff: float = M_E_EFF,
    m_h_eff: float = M_H_EFF,
) -> tuple[float, float, float]:
    nc = 2.0 * ((m_e_eff * M0 * K_B * temperature_k) / (2.0 * np.pi * HBAR**2)) ** 1.5
    nv = 2.0 * ((m_h_eff * M0 * K_B * temperature_k) / (2.0 * np.pi * HBAR**2)) ** 1.5

    delta_mass_term_ev = (K_B * temperature_k / E_CHARGE) * np.log(nc / nv)
    mu_e_ev = 0.5 * (qfls_ev - delta_mass_term_ev)
    mu_h_ev = 0.5 * (qfls_ev + delta_mass_term_ev)

    n_m3 = nc * np.exp(((mu_e_ev - eg_ev / 2.0) * E_CHARGE) / (K_B * temperature_k))
    n_cm3 = n_m3 / 1e6
    return mu_e_ev, mu_h_ev, n_cm3


def _compute_linear_fit_and_covariance(
    x_j: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, np.ndarray]:
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

    return float(slope), float(intercept), float(r2), covariance


def _compute_parameters_from_line(
    slope: float, intercept: float, assumed_a0: float
) -> tuple[float, float, float, float, float, float]:
    if (not np.isfinite(slope)) or slope == 0 or (not np.isfinite(intercept)):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    temperature_k = -1.0 / (K_B * slope)
    qfls_effective_j = intercept * K_B * temperature_k
    qfls_effective_ev = qfls_effective_j / E_CHARGE
    qfls_ev = qfls_effective_ev - (K_B * temperature_k / E_CHARGE) * np.log(assumed_a0)
    mu_e_ev, mu_h_ev, n_cm3 = compute_mu_and_density_mb(temperature_k=temperature_k, qfls_ev=qfls_ev)
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


def _fit_range_rms_uncertainty(
    energy_ev: np.ndarray,
    intensity: np.ndarray,
    base_fit_mask: np.ndarray,
    base_parameters: dict[str, float],
    assumed_a0: float,
) -> tuple[dict[str, float], int]:
    out = {key: np.nan for key in base_parameters}
    if (not ESTIMATE_FIT_RANGE_UNCERTAINTY) or FIT_RANGE_SCAN_SHIFT_POINTS < 1:
        return out, 0

    idx_base = np.flatnonzero(base_fit_mask)
    if idx_base.size < 3:
        return out, 0

    valid = np.isfinite(energy_ev) & np.isfinite(intensity) & (intensity > 0)
    lo0 = int(idx_base[0])
    hi0 = int(idx_base[-1])
    n_total = int(energy_ev.size)
    idx_grid = np.arange(n_total)
    min_points = max(3, FIT_RANGE_SCAN_MIN_POINTS)

    samples = {key: [] for key in base_parameters}
    n_windows = 0

    for dlo in range(-FIT_RANGE_SCAN_SHIFT_POINTS, FIT_RANGE_SCAN_SHIFT_POINTS + 1):
        for dhi in range(-FIT_RANGE_SCAN_SHIFT_POINTS, FIT_RANGE_SCAN_SHIFT_POINTS + 1):
            lo = lo0 + dlo
            hi = hi0 + dhi
            if lo < 0 or hi >= n_total or hi <= lo:
                continue

            mask_window = (idx_grid >= lo) & (idx_grid <= hi) & valid
            if np.count_nonzero(mask_window) < min_points:
                continue

            x_j = energy_ev[mask_window] * E_CHARGE
            y = linearized_signal(energy_ev[mask_window], intensity[mask_window])
            slope, intercept, r2, _ = _compute_linear_fit_and_covariance(x_j, y)
            (
                temperature_k,
                qfls_effective_ev,
                qfls_ev,
                mu_e_ev,
                mu_h_ev,
                n_cm3,
            ) = _compute_parameters_from_line(slope=slope, intercept=intercept, assumed_a0=assumed_a0)

            if FIT_RANGE_SCAN_REQUIRE_PHYSICAL_BOUNDS:
                if (
                    (not np.isfinite(temperature_k))
                    or (temperature_k < WINDOW_T_MIN_K)
                    or (temperature_k > WINDOW_T_MAX_K)
                    or (not np.isfinite(r2))
                    or (r2 < FIT_RANGE_SCAN_MIN_R2)
                    or (not np.isfinite(slope))
                    or (slope >= 0)
                ):
                    continue

            values = {
                "temperature_k": temperature_k,
                "qfls_effective_ev": qfls_effective_ev,
                "qfls_ev": qfls_ev,
                "mu_e_ev": mu_e_ev,
                "mu_h_ev": mu_h_ev,
                "carrier_density_cm3": n_cm3,
            }
            for key, value in values.items():
                if np.isfinite(value):
                    samples[key].append(float(value))
            n_windows += 1

    if n_windows == 0:
        return out, 0

    for key, base_value in base_parameters.items():
        sample_arr = np.array(samples[key], dtype=float)
        if sample_arr.size == 0 or (not np.isfinite(base_value)):
            out[key] = np.nan
        else:
            out[key] = float(np.sqrt(np.mean((sample_arr - base_value) ** 2)))
    return out, n_windows


def _combine_uncertainties(err_a: float, err_b: float) -> float:
    a_ok = np.isfinite(err_a)
    b_ok = np.isfinite(err_b)
    if a_ok and b_ok:
        return float(np.sqrt(err_a**2 + err_b**2))
    if a_ok:
        return float(abs(err_a))
    if b_ok:
        return float(abs(err_b))
    return np.nan


def _safe_log_yerr(y: np.ndarray, err: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    err = np.asarray(err, dtype=float)
    lower = np.where(np.isfinite(y) & np.isfinite(err) & (y > 0), np.minimum(err, 0.95 * y), np.nan)
    upper = np.where(np.isfinite(err), err, np.nan)
    return np.vstack([lower, upper])


def auto_select_fit_window(
    energy_ev: np.ndarray,
    intensity: np.ndarray,
) -> tuple[np.ndarray, float, float, str]:
    valid = np.isfinite(energy_ev) & np.isfinite(intensity) & (intensity > 0)
    if np.count_nonzero(valid) < WINDOW_MIN_POINTS:
        return valid, np.nan, np.nan, "fallback_insufficient_points"

    peak_idx = np.argmax(np.where(valid, intensity, -np.inf))
    peak_ev = float(energy_ev[peak_idx])
    search_min = max(WINDOW_SEARCH_MIN_EV, peak_ev + WINDOW_PEAK_OFFSET_EV)
    search_max = WINDOW_SEARCH_MAX_EV

    candidate = valid & (energy_ev >= search_min) & (energy_ev <= search_max)
    window_mode = "auto_peak_offset"

    if np.count_nonzero(candidate) < WINDOW_MIN_POINTS:
        candidate = valid & (energy_ev >= WINDOW_SEARCH_MIN_EV) & (energy_ev <= WINDOW_SEARCH_MAX_EV)
        window_mode = "fallback_search_range_only"

    if np.count_nonzero(candidate) < WINDOW_MIN_POINTS:
        candidate = valid
        window_mode = "fallback_all_valid"

    idx = np.flatnonzero(candidate)
    x_j = energy_ev[idx] * E_CHARGE
    y = linearized_signal(energy_ev[idx], intensity[idx])

    best: dict[str, float | int] | None = None
    n = idx.size

    for i in range(0, n - WINDOW_MIN_POINTS + 1):
        for j in range(i + WINDOW_MIN_POINTS - 1, n):
            xx = x_j[i : j + 1]
            yy = y[i : j + 1]
            slope, intercept = np.polyfit(xx, yy, deg=1)
            if not np.isfinite(slope) or slope >= 0:
                continue

            yy_fit = slope * xx + intercept
            ss_res = np.sum((yy - yy_fit) ** 2)
            ss_tot = np.sum((yy - np.mean(yy)) ** 2)
            if ss_tot <= 0:
                continue
            r2 = 1.0 - ss_res / ss_tot
            if (not np.isfinite(r2)) or (r2 < WINDOW_MIN_R2):
                continue

            temperature_k = -1.0 / (K_B * slope)
            if (not np.isfinite(temperature_k)) or (temperature_k < WINDOW_T_MIN_K) or (temperature_k > WINDOW_T_MAX_K):
                continue

            length = j - i + 1
            window_start_ev = float(energy_ev[idx[i]])
            score = (
                r2
                + WINDOW_LENGTH_WEIGHT * length
                + WINDOW_HIGH_ENERGY_WEIGHT * (window_start_ev - float(np.nanmin(energy_ev)))
            )

            if (best is None) or (score > float(best["score"])):
                best = {"score": score, "i": i, "j": j}

    if best is None:
        i = 0
        j = n - 1
        window_mode += "|fallback_full_candidate"
    else:
        i = int(best["i"])
        j = int(best["j"])

    selected_idx = idx[i : j + 1]
    fit_mask = np.zeros_like(valid, dtype=bool)
    fit_mask[selected_idx] = True
    fit_min_ev = float(energy_ev[selected_idx[0]])
    fit_max_ev = float(energy_ev[selected_idx[-1]])
    return fit_mask, fit_min_ev, fit_max_ev, window_mode


def fit_single_spectrum(
    energy_ev: np.ndarray,
    intensity: np.ndarray,
    spectrum_id: str,
    intensity_w_cm2: float,
    auto_select_fit_window_enabled: bool,
    fit_min_ev_fixed: float,
    fit_max_ev_fixed: float,
    assumed_a0: float,
) -> tuple[FitResult, np.ndarray]:
    if auto_select_fit_window_enabled:
        fit_mask, fit_min_ev, fit_max_ev, window_mode = auto_select_fit_window(energy_ev=energy_ev, intensity=intensity)
    else:
        valid = np.isfinite(energy_ev) & np.isfinite(intensity) & (intensity > 0)
        in_window = (energy_ev >= fit_min_ev_fixed) & (energy_ev <= fit_max_ev_fixed)
        fit_mask = valid & in_window
        fit_min_ev = fit_min_ev_fixed
        fit_max_ev = fit_max_ev_fixed
        window_mode = "fixed"

    if np.count_nonzero(fit_mask) < 3:
        result = FitResult(
            spectrum_id=spectrum_id,
            intensity_w_cm2=float(intensity_w_cm2),
            fit_min_ev=float(fit_min_ev),
            fit_max_ev=float(fit_max_ev),
            window_mode=window_mode,
            n_points_fit=int(np.count_nonzero(fit_mask)),
            slope=np.nan,
            intercept=np.nan,
            r2=np.nan,
            temperature_k=np.nan,
            qfls_effective_ev=np.nan,
            qfls_ev=np.nan,
            mu_e_ev=np.nan,
            mu_h_ev=np.nan,
            carrier_density_cm3=np.nan,
            temperature_err_chi2_k=np.nan,
            temperature_err_range_k=np.nan,
            temperature_err_total_k=np.nan,
            qfls_effective_err_chi2_ev=np.nan,
            qfls_effective_err_range_ev=np.nan,
            qfls_effective_err_total_ev=np.nan,
            qfls_err_chi2_ev=np.nan,
            qfls_err_range_ev=np.nan,
            qfls_err_total_ev=np.nan,
            mu_e_err_chi2_ev=np.nan,
            mu_e_err_range_ev=np.nan,
            mu_e_err_total_ev=np.nan,
            mu_h_err_chi2_ev=np.nan,
            mu_h_err_range_ev=np.nan,
            mu_h_err_total_ev=np.nan,
            carrier_density_err_chi2_cm3=np.nan,
            carrier_density_err_range_cm3=np.nan,
            carrier_density_err_total_cm3=np.nan,
            fit_range_samples=0,
        )
        return result, np.full_like(intensity, np.nan, dtype=float)

    x_j = energy_ev[fit_mask] * E_CHARGE
    y = linearized_signal(energy_ev[fit_mask], intensity[fit_mask])
    slope, intercept, r2, covariance_line = _compute_linear_fit_and_covariance(x_j, y)
    temperature_k, qfls_effective_ev, qfls_ev, mu_e_ev, mu_h_ev, n_cm3 = _compute_parameters_from_line(
        slope=slope, intercept=intercept, assumed_a0=assumed_a0
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
    range_err, n_windows_range = _fit_range_rms_uncertainty(
        energy_ev=energy_ev,
        intensity=intensity,
        base_fit_mask=fit_mask,
        base_parameters=base_parameters,
        assumed_a0=assumed_a0,
    )

    temperature_err_total_k = _combine_uncertainties(chi2_err["temperature_k"], range_err["temperature_k"])
    qfls_effective_err_total_ev = _combine_uncertainties(
        chi2_err["qfls_effective_ev"], range_err["qfls_effective_ev"]
    )
    qfls_err_total_ev = _combine_uncertainties(chi2_err["qfls_ev"], range_err["qfls_ev"])
    mu_e_err_total_ev = _combine_uncertainties(chi2_err["mu_e_ev"], range_err["mu_e_ev"])
    mu_h_err_total_ev = _combine_uncertainties(chi2_err["mu_h_ev"], range_err["mu_h_ev"])
    carrier_density_err_total_cm3 = _combine_uncertainties(
        chi2_err["carrier_density_cm3"], range_err["carrier_density_cm3"]
    )

    if np.isfinite(temperature_k) and temperature_k > 0 and np.isfinite(qfls_effective_ev):
        qfls_effective_j = qfls_effective_ev * E_CHARGE
        energy_j_all = energy_ev * E_CHARGE
        ln_prefactor = np.log(2.0 * energy_j_all**2 / (H**3 * C**2))
        ln_i_model = ln_prefactor - (energy_j_all - qfls_effective_j) / (K_B * temperature_k)
        intensity_model = np.exp(ln_i_model)
    else:
        intensity_model = np.full_like(intensity, np.nan, dtype=float)

    result = FitResult(
        spectrum_id=spectrum_id,
        intensity_w_cm2=float(intensity_w_cm2),
        fit_min_ev=float(fit_min_ev),
        fit_max_ev=float(fit_max_ev),
        window_mode=window_mode,
        n_points_fit=int(np.count_nonzero(fit_mask)),
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        temperature_k=float(temperature_k),
        qfls_effective_ev=float(qfls_effective_ev),
        qfls_ev=float(qfls_ev),
        mu_e_ev=float(mu_e_ev),
        mu_h_ev=float(mu_h_ev),
        carrier_density_cm3=float(n_cm3),
        temperature_err_chi2_k=float(chi2_err["temperature_k"]),
        temperature_err_range_k=float(range_err["temperature_k"]),
        temperature_err_total_k=float(temperature_err_total_k),
        qfls_effective_err_chi2_ev=float(chi2_err["qfls_effective_ev"]),
        qfls_effective_err_range_ev=float(range_err["qfls_effective_ev"]),
        qfls_effective_err_total_ev=float(qfls_effective_err_total_ev),
        qfls_err_chi2_ev=float(chi2_err["qfls_ev"]),
        qfls_err_range_ev=float(range_err["qfls_ev"]),
        qfls_err_total_ev=float(qfls_err_total_ev),
        mu_e_err_chi2_ev=float(chi2_err["mu_e_ev"]),
        mu_e_err_range_ev=float(range_err["mu_e_ev"]),
        mu_e_err_total_ev=float(mu_e_err_total_ev),
        mu_h_err_chi2_ev=float(chi2_err["mu_h_ev"]),
        mu_h_err_range_ev=float(range_err["mu_h_ev"]),
        mu_h_err_total_ev=float(mu_h_err_total_ev),
        carrier_density_err_chi2_cm3=float(chi2_err["carrier_density_cm3"]),
        carrier_density_err_range_cm3=float(range_err["carrier_density_cm3"]),
        carrier_density_err_total_cm3=float(carrier_density_err_total_cm3),
        fit_range_samples=int(n_windows_range),
    )
    return result, intensity_model


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
        ax.plot(energy_ev, spectra[:, i], color=cmap(norm(intensities_w_cm2[i])), lw=1.15, alpha=0.96)

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

    ax0.plot(energy_ev, intensity, color="#1f4e79", lw=1.8, label="Experiment")
    ax0.plot(energy_ev, intensity_model, color="#d32f2f", lw=1.45, ls="--", label="High-energy GPL fit")
    ax0.axvspan(fit_min_ev, fit_max_ev, color="0.65", alpha=0.18, label="Selected fit window")
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
    )
    ax0.text(
        0.985,
        0.97,
        info_text,
        transform=ax0.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.3", "boxstyle": "square,pad=0.25", "alpha": 0.95},
    )

    y_all = linearized_signal(energy_ev[intensity > 0], intensity[intensity > 0])
    ax1.plot(energy_ev[intensity > 0], y_all, color="0.35", lw=1.05, label="Linearized data")

    x_fit_ev = energy_ev[fit_mask]
    x_fit_j = x_fit_ev * E_CHARGE
    y_line = result.slope * x_fit_j + result.intercept
    y_fit_data = linearized_signal(x_fit_ev, intensity[fit_mask])
    ax1.scatter(x_fit_ev, y_fit_data, s=13, color="#2e7d32", alpha=0.8, zorder=3, label="Points used for fit")
    ax1.plot(x_fit_ev, y_line, color="#d32f2f", lw=1.5, ls="-", label="Linear regression")
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
        label=r"$\mu_e$",
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
        label=r"$\mu_h$",
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
    )
    style_axes(ax11, logx=True, logy=True)
    ax11.set_xlabel(r"Excitation intensity, $I_{exc}$ (W cm$^{-2}$)")
    ax11.set_ylabel(r"Carrier density, $n$ (cm$^{-3}$)")
    ax11.text(0.03, 0.93, "(d)", transform=ax11.transAxes, fontsize=11, fontweight="semibold")

    if np.isfinite(x_min_plot) and np.isfinite(x_max_plot) and (x_max_plot > x_min_plot):
        for ax in (ax00, ax01, ax10, ax11):
            ax.set_xlim(x_min_plot, x_max_plot)

    fig.suptitle("Extracted hot-carrier parameters versus excitation intensity", y=1.01, fontsize=13)
    fig.tight_layout(pad=0.7)
    save_figure(fig, outpath)
    plt.close(fig)


def main() -> None:
    setup_plot_style()
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
            f"Intensity list has {len(EXCITATION_INTENSITY_W_CM2)} values but file has {spectra.shape[1]} spectra."
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

    # First spectrum fit (requested starting point)
    first_result, first_model = fit_single_spectrum(
        energy_ev=energy_ev,
        intensity=spectra[:, 0],
        spectrum_id=spectrum_ids[0],
        intensity_w_cm2=float(EXCITATION_INTENSITY_W_CM2[0]),
        auto_select_fit_window_enabled=AUTO_SELECT_FIT_WINDOW,
        fit_min_ev_fixed=FIT_ENERGY_MIN_EV,
        fit_max_ev_fixed=FIT_ENERGY_MAX_EV,
        assumed_a0=ASSUMED_A0,
    )
    plot_single_fit(
        energy_ev=energy_ev,
        intensity=spectra[:, 0],
        intensity_model=first_model,
        result=first_result,
        outpath=fit_dir / "fit_spectrum_00.png",
    )

    all_results: list[FitResult] = [first_result]

    # Then iterate over all remaining spectra
    for i in range(1, spectra.shape[1]):
        result, intensity_model = fit_single_spectrum(
            energy_ev=energy_ev,
            intensity=spectra[:, i],
            spectrum_id=spectrum_ids[i],
            intensity_w_cm2=float(EXCITATION_INTENSITY_W_CM2[i]),
            auto_select_fit_window_enabled=AUTO_SELECT_FIT_WINDOW,
            fit_min_ev_fixed=FIT_ENERGY_MIN_EV,
            fit_max_ev_fixed=FIT_ENERGY_MAX_EV,
            assumed_a0=ASSUMED_A0,
        )
        all_results.append(result)
        plot_single_fit(
            energy_ev=energy_ev,
            intensity=spectra[:, i],
            intensity_model=intensity_model,
            result=result,
            outpath=fit_dir / f"fit_spectrum_{i:02d}.png",
        )

    results_df = pd.DataFrame([r.__dict__ for r in all_results])
    results_df.to_csv(out_dir / "fit_results.csv", index=False)
    plot_summary(results_df, out_dir / "parameters_vs_intensity.png")

    print("Done.")
    print(f"Raw spectra plot: {out_dir / 'all_spectra_logscale.png'}")
    print(f"Spectrum fits:    {fit_dir}")
    print(f"Results table:    {out_dir / 'fit_results.csv'}")
    print(f"Summary figure:   {out_dir / 'parameters_vs_intensity.png'}")
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
            "Uncertainty mode: chi^2 + fit-range RMS | "
            f"shift_points={FIT_RANGE_SCAN_SHIFT_POINTS}, "
            f"min_points={FIT_RANGE_SCAN_MIN_POINTS}, "
            f"min_r2={FIT_RANGE_SCAN_MIN_R2:.3f}"
        )
    else:
        print("Uncertainty mode: chi^2 only")


if __name__ == "__main__":
    main()
