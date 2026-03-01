from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ABSORPTIVITY_AT_LASER,
    ABSORPTIVITY_AT_LASER_SIGMA,
    ACTIVE_LAYER_THICKNESS_NM,
    ASSUMED_A0,
    A0_SIGMA,
    EG_EV,
    ESTIMATE_FIT_RANGE_UNCERTAINTY,
    E_CHARGE,
    FD_BISECTION_ETA_BOUND,
    FD_BISECTION_MAX_ITER,
    FD_BISECTION_TOL,
    FD_F12_MIDPOINTS,
    FIT_RANGE_SCAN_MIN_POINTS,
    FIT_RANGE_SCAN_MIN_R2,
    FIT_RANGE_SCAN_MAX_WINDOWS_TO_PLOT,
    FIT_RANGE_SCAN_PLOT_WEIGHT_COVERAGE,
    FIT_RANGE_SCAN_REQUIRE_PHYSICAL_BOUNDS,
    K_B,
    H,
    C,
    HBAR,
    LASER_WAVELENGTH_NM,
    M0,
    M_E_EFF,
    M_H_EFF,
    MB_VALIDITY_REFERENCE_TEMPERATURES_K,
    MB_VALIDITY_REL_ERROR_LIMIT,
    MB_VALIDITY_X_MAX,
    MB_VALIDITY_X_MIN,
    MB_VALIDITY_X_POINTS,
    PLQY_ETA,
    PLQY_ETA_SIGMA,
    WINDOW_MIN_POINTS,
    WINDOW_MIN_R2,
    WINDOW_PEAK_OFFSET_EV,
    WINDOW_SEARCH_MAX_EV,
    WINDOW_SEARCH_MIN_EV,
    WINDOW_T_MAX_K,
    WINDOW_T_MIN_K,
)
from .models import (
    FIT_PARAMETER_KEYS,
    FitResult,
    WindowFitSample,
    _build_fit_result,
    _build_intensity_model,
    _combine_parameter_error_components,
    _empty_like_fit_result,
)


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


def _broadcast_parameter_vector(
    value: float | np.ndarray,
    size: int,
    parameter_name: str,
) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(size, float(arr), dtype=float)
    arr = arr.ravel()
    if arr.size != size:
        raise ValueError(
            f"{parameter_name} must be scalar or length {size}, got length {arr.size}."
        )
    return arr.astype(float)


def _polylog23_series(
    r: np.ndarray,
    *,
    rtol: float = 1e-12,
    max_terms: int = 120000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate Li_2(r) and Li_3(r) from power series for |r| < 1:
      Li_s(r) = sum_{m=1..infinity} r^m / m^s.
    """
    r_arr = np.asarray(r, dtype=float)
    li2 = np.full_like(r_arr, np.nan, dtype=float)
    li3 = np.full_like(r_arr, np.nan, dtype=float)
    valid = np.isfinite(r_arr) & (r_arr >= 0.0) & (r_arr < 1.0)
    if not np.any(valid):
        return li2, li3

    r_valid = r_arr[valid]
    power = r_valid.copy()
    li2_valid = power.copy()
    li3_valid = power.copy()

    n = 1
    while n < max_terms:
        n += 1
        power *= r_valid
        inv_n = 1.0 / n
        term2 = power * (inv_n**2)
        term3 = term2 * inv_n
        li2_valid += term2
        li3_valid += term3
        if np.all(
            np.abs(term2) <= rtol * np.maximum(1.0, np.abs(li2_valid))
        ) and np.all(
            np.abs(term3) <= rtol * np.maximum(1.0, np.abs(li3_valid))
        ):
            break

    li2[valid] = li2_valid
    li3[valid] = li3_valid
    return li2, li3


def integrated_ipc_step_absorber_be(
    temperature_k: float,
    reduced_qfls: np.ndarray,
    *,
    eg_ev: float = EG_EV,
    absorptivity_a0: float = ASSUMED_A0,
) -> np.ndarray:
    """
    Exact GPL integral for A(E)=A0*Theta(E-Eg):
      I_PC(E) = [2E^2/(h^3 c^2)] * A0 / (exp((E-Delta_mu)/(k_B T)) - 1)
      Phi_BE = integral_{Eg..infinity} I_PC(E) dE

    With x=(Delta_mu-Eg)/(k_B T), r=exp(x), and x<0:
      Phi_BE = [2A0/(h^3 c^2)] *
               [Eg^2(k_B T) Li_1(r) + 2Eg(k_B T)^2 Li_2(r) + 2(k_B T)^3 Li_3(r)].
    """
    x = np.asarray(reduced_qfls, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    if (
        (not np.isfinite(temperature_k))
        or (temperature_k <= 0)
        or (not np.isfinite(eg_ev))
        or (eg_ev <= 0)
        or (not np.isfinite(absorptivity_a0))
        or (absorptivity_a0 <= 0)
    ):
        return out

    valid = np.isfinite(x) & (x < 0.0)
    if not np.any(valid):
        return out

    x_valid = x[valid]
    r = np.exp(np.clip(x_valid, -700.0, -1e-12))
    li1 = -np.log1p(-r)  # Li_1(r)
    li2, li3 = _polylog23_series(r)
    kbt = K_B * float(temperature_k)
    eg_j = float(eg_ev) * E_CHARGE
    prefactor = (2.0 * float(absorptivity_a0)) / (H**3 * C**2)
    phi_be = prefactor * (
        (eg_j**2) * kbt * li1
        + 2.0 * eg_j * (kbt**2) * li2
        + 2.0 * (kbt**3) * li3
    )
    out[valid] = phi_be
    return out


def integrated_ipc_step_absorber_mb(
    temperature_k: float,
    reduced_qfls: np.ndarray,
    *,
    eg_ev: float = EG_EV,
    absorptivity_a0: float = ASSUMED_A0,
) -> np.ndarray:
    """
    MB limit of the same integral (first Boltzmann term only):
      Phi_MB = [2A0/(h^3 c^2)] * exp(x) *
               [Eg^2(k_B T) + 2Eg(k_B T)^2 + 2(k_B T)^3],
    where x=(Delta_mu-Eg)/(k_B T).
    """
    x = np.asarray(reduced_qfls, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    if (
        (not np.isfinite(temperature_k))
        or (temperature_k <= 0)
        or (not np.isfinite(eg_ev))
        or (eg_ev <= 0)
        or (not np.isfinite(absorptivity_a0))
        or (absorptivity_a0 <= 0)
    ):
        return out

    valid = np.isfinite(x)
    if not np.any(valid):
        return out

    x_valid = x[valid]
    kbt = K_B * float(temperature_k)
    eg_j = float(eg_ev) * E_CHARGE
    prefactor = (2.0 * float(absorptivity_a0)) / (H**3 * C**2)
    phi_mb = prefactor * np.exp(np.clip(x_valid, -700.0, 700.0)) * (
        (eg_j**2) * kbt + 2.0 * eg_j * (kbt**2) + 2.0 * (kbt**3)
    )
    out[valid] = phi_mb
    return out


def build_mb_validity_scan(
    temperatures_k: np.ndarray,
    *,
    x_min: float = MB_VALIDITY_X_MIN,
    x_max: float = MB_VALIDITY_X_MAX,
    x_points: int = MB_VALIDITY_X_POINTS,
    rel_error_limit: float = MB_VALIDITY_REL_ERROR_LIMIT,
    eg_ev: float = EG_EV,
    absorptivity_a0: float = ASSUMED_A0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build MB-validity curves:
      y = ln[integral_{Eg..infinity} I_PC(E) dE] vs x=(Delta_mu-Eg)/(k_B T)
    using exact BE photons and MB approximation.
    """
    temp_arr = np.asarray(temperatures_k, dtype=float)
    temp_arr = temp_arr[np.isfinite(temp_arr) & (temp_arr > 0)]
    if temp_arr.size == 0:
        temp_arr = np.asarray(MB_VALIDITY_REFERENCE_TEMPERATURES_K, dtype=float)
        temp_arr = temp_arr[np.isfinite(temp_arr) & (temp_arr > 0)]
    if temp_arr.size == 0:
        return pd.DataFrame(), pd.DataFrame()

    x_lo = float(min(x_min, x_max))
    x_hi = float(max(x_min, x_max))
    x_hi = min(x_hi, -1e-6)
    if x_lo >= x_hi:
        x_lo, x_hi = -12.0, -0.05
    x_grid = np.linspace(x_lo, x_hi, max(60, int(x_points)), dtype=float)

    curve_frames: list[pd.DataFrame] = []
    limit_records: list[dict[str, float | bool]] = []
    unique_temps = np.unique(np.round(temp_arr, 6))

    for temperature_k in unique_temps:
        kbt_ev = (K_B * temperature_k) / E_CHARGE
        delta_mu_ev = eg_ev + x_grid * kbt_ev
        phi_be = integrated_ipc_step_absorber_be(
            temperature_k=temperature_k,
            reduced_qfls=x_grid,
            eg_ev=eg_ev,
            absorptivity_a0=absorptivity_a0,
        )
        phi_mb = integrated_ipc_step_absorber_mb(
            temperature_k=temperature_k,
            reduced_qfls=x_grid,
            eg_ev=eg_ev,
            absorptivity_a0=absorptivity_a0,
        )

        ln_phi_be = np.where(phi_be > 0, np.log(phi_be), np.nan)
        ln_phi_mb = np.where(phi_mb > 0, np.log(phi_mb), np.nan)
        rel_error = np.where(
            (phi_be > 0) & (phi_mb > 0),
            (phi_be / phi_mb) - 1.0,
            np.nan,
        )
        log_deviation = ln_phi_be - ln_phi_mb
        is_mb_valid = np.isfinite(rel_error) & (rel_error <= rel_error_limit)

        curve_frames.append(
            pd.DataFrame(
                {
                    "temperature_k": float(temperature_k),
                    "reduced_qfls": x_grid,
                    "delta_mu_ev": delta_mu_ev,
                    "integral_ipc_be": phi_be,
                    "integral_ipc_mb": phi_mb,
                    "ln_integral_ipc_be": ln_phi_be,
                    "ln_integral_ipc_mb": ln_phi_mb,
                    "mb_relative_error": rel_error,
                    "mb_log_deviation": log_deviation,
                    "mb_valid": is_mb_valid,
                }
            )
        )

        invalid_idx = np.flatnonzero(
            np.isfinite(rel_error) & (rel_error > rel_error_limit)
        )
        x_limit = float(x_grid[invalid_idx[0]]) if invalid_idx.size > 0 else np.nan
        delta_mu_limit_ev = (
            float(eg_ev + x_limit * kbt_ev) if np.isfinite(x_limit) else np.nan
        )
        limit_records.append(
            {
                "temperature_k": float(temperature_k),
                "x_limit": x_limit,
                "delta_mu_limit_ev": delta_mu_limit_ev,
                "mb_relative_error_limit": float(rel_error_limit),
                "mb_valid_over_entire_scan": bool(invalid_idx.size == 0),
            }
        )

    curves_df = pd.concat(curve_frames, ignore_index=True)
    limits_df = pd.DataFrame.from_records(limit_records)
    finite_limits = limits_df["x_limit"].to_numpy(dtype=float)
    finite_limits = finite_limits[np.isfinite(finite_limits)]
    conservative_x = float(np.min(finite_limits)) if finite_limits.size > 0 else np.nan
    limits_df["x_limit_conservative"] = conservative_x
    return curves_df, limits_df


def compute_power_balance_table(
    results_df: pd.DataFrame,
    laser_wavelength_nm: float = LASER_WAVELENGTH_NM,
    absorptivity_at_laser: float = ABSORPTIVITY_AT_LASER,
    absorptivity_at_laser_sigma: float = ABSORPTIVITY_AT_LASER_SIGMA,
    plqy_eta: float | np.ndarray = PLQY_ETA,
    plqy_eta_sigma: float | np.ndarray = PLQY_ETA_SIGMA,
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
    n_rows = p_exc_w_cm2.size
    plqy_eta_arr = _broadcast_parameter_vector(
        value=plqy_eta,
        size=n_rows,
        parameter_name="plqy_eta",
    )
    plqy_eta_sigma_arr = _broadcast_parameter_vector(
        value=plqy_eta_sigma,
        size=n_rows,
        parameter_name="plqy_eta_sigma",
    )
    if np.any((~np.isfinite(plqy_eta_arr)) | (plqy_eta_arr < 0) | (plqy_eta_arr > 1)):
        raise ValueError("plqy_eta values must be finite and within [0, 1].")
    if np.any((~np.isfinite(plqy_eta_sigma_arr)) | (plqy_eta_sigma_arr < 0)):
        raise ValueError("plqy_eta_sigma values must be finite and non-negative.")

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
    phi_rad_cm2_s = plqy_eta_arr * phi_abs_cm2_s
    phi_nonrad_cm2_s = (1.0 - plqy_eta_arr) * phi_abs_cm2_s

    d_phi_rad_d_a = plqy_eta_arr * p_exc_w_cm2 / e_laser_j
    d_phi_rad_d_eta = phi_abs_cm2_s
    phi_rad_err_cm2_s = np.sqrt(
        (d_phi_rad_d_a * absorptivity_at_laser_sigma) ** 2
        + (d_phi_rad_d_eta * plqy_eta_sigma_arr) ** 2
    )

    d_phi_nonrad_d_a = (1.0 - plqy_eta_arr) * p_exc_w_cm2 / e_laser_j
    d_phi_nonrad_d_eta = -phi_abs_cm2_s
    phi_nonrad_err_cm2_s = np.sqrt(
        (d_phi_nonrad_d_a * absorptivity_at_laser_sigma) ** 2
        + (d_phi_nonrad_d_eta * plqy_eta_sigma_arr) ** 2
    )

    valid_temperature = np.isfinite(temperature_k) & (temperature_k > 0)
    beta_j = np.where(
        valid_temperature,
        eg_j + (3.0 - 2.0 * plqy_eta_arr) * K_B * temperature_k,
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
    d_p_rec_d_eta = np.where(
        valid_temperature,
        prefactor * (-2.0 * K_B * temperature_k),
        np.nan,
    )
    d_p_rec_d_t = np.where(
        valid_temperature,
        prefactor * (3.0 - 2.0 * plqy_eta_arr) * K_B,
        np.nan,
    )
    p_rec_err_w_cm2 = np.sqrt(
        (d_p_rec_d_a * absorptivity_at_laser_sigma) ** 2
        + (d_p_rec_d_eta * plqy_eta_sigma_arr) ** 2
        + (d_p_rec_d_t * temperature_err_k) ** 2
    )

    d_p_th_d_a = np.where(valid_temperature, p_exc_w_cm2 * cooling_factor, np.nan)
    d_p_th_d_eta = np.where(valid_temperature, prefactor * (2.0 * K_B * temperature_k), np.nan)
    d_p_th_d_t = np.where(
        valid_temperature,
        -prefactor * (3.0 - 2.0 * plqy_eta_arr) * K_B,
        np.nan,
    )
    p_th_err_w_cm2 = np.sqrt(
        (d_p_th_d_a * absorptivity_at_laser_sigma) ** 2
        + (d_p_th_d_eta * plqy_eta_sigma_arr) ** 2
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
            "plqy_eta": plqy_eta_arr,
            "plqy_eta_sigma": plqy_eta_sigma_arr,
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
