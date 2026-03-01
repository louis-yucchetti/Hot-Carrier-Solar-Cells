from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from .config import (
    E_CHARGE,
    EG_EV,
    HBAR,
    K_B,
    M0,
    M_E_EFF,
    TSAI_ENABLE_SIMULATION,
    TSAI_EPSILON_INF,
    TSAI_EPSILON_STATIC,
    TSAI_LATTICE_TEMPERATURE_K,
    TSAI_LO_PHONON_ENERGY_EV,
    TSAI_LO_PHONON_LIFETIME_PS,
    TSAI_MU_E_GRID_MARGIN_EV,
    TSAI_MU_E_GRID_MAX_EV,
    TSAI_MU_E_GRID_MIN_EV,
    TSAI_MU_E_GRID_POINTS,
    TSAI_PTH_INVERSE_POINTS,
    TSAI_Q_MAX_CM1,
    TSAI_Q_MIN_CM1,
    TSAI_Q_POINTS,
    TSAI_SCREENING_DMU_EV,
    TSAI_SCREENING_FD_MIDPOINTS,
    TSAI_SCREENING_MODEL,
    TSAI_T_GRID_MAX_K,
    TSAI_T_GRID_MIN_K,
    TSAI_T_GRID_POINTS,
    TSAI_USE_STATIC_SCREENING,
)

# Vacuum permittivity (F m^-1)
EPSILON_0 = 8.8541878128e-12


@dataclass
class TsaiWorkflowResult:
    forward_table_df: pd.DataFrame
    inverse_table_df: pd.DataFrame
    experimental_prediction_df: pd.DataFrame
    mu_grid_ev: np.ndarray
    temperature_grid_k: np.ndarray
    p_th_grid_w_cm3: np.ndarray


def _log1p_exp(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x > 50.0, x, np.log1p(np.exp(x)))


def _fermi_dirac_half(eta: float, midpoints: int) -> float:
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

    n_mid = max(80, int(midpoints))
    x_max = max(40.0, eta + 40.0)
    x = np.linspace(0.0, x_max, n_mid, dtype=float)
    u = x - eta
    occupation = np.where(u > 40.0, np.exp(-u), 1.0 / (1.0 + np.exp(u)))
    integrand = np.sqrt(x) * occupation
    return float((2.0 / np.sqrt(np.pi)) * np.trapezoid(integrand, x))


def _effective_density_of_states_electron_m3(
    temperature_k: float,
    electron_mass_eff: float = M_E_EFF,
) -> float:
    if (not np.isfinite(temperature_k)) or (temperature_k <= 0):
        return np.nan
    return float(
        2.0
        * ((electron_mass_eff * M0 * K_B * temperature_k) / (2.0 * np.pi * HBAR**2))
        ** 1.5
    )


def _electron_density_fd_m3(
    mu_c_j: float,
    temperature_k: float,
    fd_midpoints: int,
    electron_mass_eff: float = M_E_EFF,
) -> float:
    if (
        (not np.isfinite(mu_c_j))
        or (not np.isfinite(temperature_k))
        or (temperature_k <= 0)
    ):
        return np.nan

    n_c = _effective_density_of_states_electron_m3(
        temperature_k=temperature_k,
        electron_mass_eff=electron_mass_eff,
    )
    if (not np.isfinite(n_c)) or (n_c <= 0):
        return np.nan

    eta = mu_c_j / (K_B * temperature_k)
    return float(n_c * _fermi_dirac_half(eta, midpoints=fd_midpoints))


def _electron_density_mb_m3(
    mu_c_j: float,
    temperature_k: float,
    electron_mass_eff: float = M_E_EFF,
) -> float:
    if (
        (not np.isfinite(mu_c_j))
        or (not np.isfinite(temperature_k))
        or (temperature_k <= 0)
    ):
        return np.nan

    n_c = _effective_density_of_states_electron_m3(
        temperature_k=temperature_k,
        electron_mass_eff=electron_mass_eff,
    )
    if (not np.isfinite(n_c)) or (n_c <= 0):
        return np.nan

    return float(n_c * np.exp(mu_c_j / (K_B * temperature_k)))


def _screening_wave_number_m1(
    mu_c_j: float,
    temperature_k: float,
    screening_model: str,
    dmu_ev: float,
    fd_midpoints: int,
    epsilon_static: float,
    use_static_screening: bool,
) -> float:
    if not use_static_screening:
        return 0.0
    if (
        (not np.isfinite(mu_c_j))
        or (not np.isfinite(temperature_k))
        or (temperature_k <= 0)
        or (epsilon_static <= 0)
    ):
        return np.nan

    if screening_model == "mb":
        n_m3 = _electron_density_mb_m3(mu_c_j=mu_c_j, temperature_k=temperature_k)
        if (not np.isfinite(n_m3)) or (n_m3 <= 0):
            return np.nan
        dn_dmu = n_m3 / (K_B * temperature_k)
    elif screening_model == "fd_fdiff":
        dmu_j = max(abs(dmu_ev) * E_CHARGE, 1e-5 * K_B * temperature_k)
        n_plus = _electron_density_fd_m3(
            mu_c_j=mu_c_j + dmu_j,
            temperature_k=temperature_k,
            fd_midpoints=fd_midpoints,
        )
        n_minus = _electron_density_fd_m3(
            mu_c_j=mu_c_j - dmu_j,
            temperature_k=temperature_k,
            fd_midpoints=fd_midpoints,
        )
        if (not np.isfinite(n_plus)) or (not np.isfinite(n_minus)):
            return np.nan
        dn_dmu = max((n_plus - n_minus) / (2.0 * dmu_j), 0.0)
    else:
        raise ValueError("TSAI_SCREENING_MODEL must be 'fd_fdiff' or 'mb'.")

    if (not np.isfinite(dn_dmu)) or (dn_dmu <= 0):
        return 0.0

    q_s_sq = (E_CHARGE**2 * dn_dmu) / (EPSILON_0 * epsilon_static)
    return float(np.sqrt(max(q_s_sq, 0.0)))


def _phonon_occupation(phonon_energy_j: float, temperature_k: float) -> float:
    if (not np.isfinite(temperature_k)) or (temperature_k <= 0):
        return np.nan
    x = phonon_energy_j / (K_B * temperature_k)
    if x > 700:
        return 0.0
    return float(1.0 / np.expm1(x))


def _screened_matrix_element_sq_times_volume_j2(
    q_m1: np.ndarray,
    q_s_m1: float,
    lo_phonon_energy_j: float,
    epsilon_inf: float,
    epsilon_static: float,
    use_static_screening: bool,
) -> np.ndarray:
    q = np.asarray(q_m1, dtype=float)
    q2 = np.maximum(q**2, np.finfo(float).tiny)

    dielectric_term = (1.0 / epsilon_inf) - (1.0 / epsilon_static)
    base = (
        (E_CHARGE**2 * lo_phonon_energy_j)
        / (2.0 * EPSILON_0 * q2)
        * dielectric_term
    )

    if not use_static_screening:
        return base

    q_s_sq = max(float(q_s_m1), 0.0) ** 2
    screening_factor = 1.0 / (1.0 + (q_s_sq / q2)) ** 2
    return base * screening_factor


def _tau_c_lo_q_seconds(
    q_m1: np.ndarray,
    mu_e_ev: float,
    temperature_k: float,
) -> tuple[np.ndarray, float]:
    """
    Eq. 41 in Tsai 2018:
      1 / tau_{c-LO}^q =
        [m_c^2 k_B T_c |M_screen^q|^2 V_c / (pi hbar^5 q)] *
        ln[(exp(eta_c - eps_min + eps_LO)+1)/(exp(eta_c - eps_min)+1)]
    """
    q = np.asarray(q_m1, dtype=float)
    q = np.where(q > 0.0, q, np.nan)
    if (not np.isfinite(mu_e_ev)) or (not np.isfinite(temperature_k)) or (temperature_k <= 0):
        return np.full_like(q, np.nan, dtype=float), np.nan

    m_c = M_E_EFF * M0
    lo_phonon_energy_j = TSAI_LO_PHONON_ENERGY_EV * E_CHARGE
    omega_lo = lo_phonon_energy_j / HBAR

    mu_c_ev = mu_e_ev - 0.5 * EG_EV
    mu_c_j = mu_c_ev * E_CHARGE
    eta_c = mu_c_j / (K_B * temperature_k)
    eps_lo = lo_phonon_energy_j / (K_B * temperature_k)

    q_s_m1 = _screening_wave_number_m1(
        mu_c_j=mu_c_j,
        temperature_k=temperature_k,
        screening_model=TSAI_SCREENING_MODEL,
        dmu_ev=TSAI_SCREENING_DMU_EV,
        fd_midpoints=TSAI_SCREENING_FD_MIDPOINTS,
        epsilon_static=TSAI_EPSILON_STATIC,
        use_static_screening=TSAI_USE_STATIC_SCREENING,
    )

    # k_min = q/2 + m_c * hbar*omega_LO / (hbar^2 q) = q/2 + m_c*omega_LO/(hbar*q)
    k_min = 0.5 * q + (m_c * omega_lo) / (HBAR * q)
    eps_min = (HBAR**2 * k_min**2) / (2.0 * m_c * K_B * temperature_k)

    log_term = _log1p_exp(eta_c - eps_min + eps_lo) - _log1p_exp(eta_c - eps_min)
    m_sq = _screened_matrix_element_sq_times_volume_j2(
        q_m1=q,
        q_s_m1=q_s_m1,
        lo_phonon_energy_j=lo_phonon_energy_j,
        epsilon_inf=TSAI_EPSILON_INF,
        epsilon_static=TSAI_EPSILON_STATIC,
        use_static_screening=TSAI_USE_STATIC_SCREENING,
    )
    inv_tau = (m_c**2 * K_B * temperature_k * m_sq) / (np.pi * HBAR**5 * q) * log_term
    inv_tau = np.where(np.isfinite(inv_tau) & (inv_tau > 0.0), inv_tau, 0.0)

    tau_q = np.full_like(inv_tau, np.inf, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        np.divide(1.0, inv_tau, out=tau_q, where=inv_tau > 0.0)
    return tau_q, q_s_m1


def compute_du_dt_intra_electron_w_cm3(
    mu_e_ev: float,
    temperature_k: float,
    q_grid_m1: np.ndarray,
    lattice_temperature_k: float = TSAI_LATTICE_TEMPERATURE_K,
) -> tuple[float, float, float]:
    """
    Eq. 48 in Tsai 2018:
      (du/dt)_intra = -(1/(2*pi^2)) * int dq q^2 hbar*omega_LO *
                      [N_q(Tc)-N_q(TL)] / [tau_{c-LO}^q + tau_LO]

    Returns:
      du_dt_intra_w_cm3, qs_cm^-1, tau_energy_s
    """
    if (
        (not np.isfinite(mu_e_ev))
        or (not np.isfinite(temperature_k))
        or (temperature_k <= 0)
    ):
        return np.nan, np.nan, np.nan

    q = np.asarray(q_grid_m1, dtype=float)
    if q.size < 2:
        return np.nan, np.nan, np.nan

    lo_phonon_energy_j = TSAI_LO_PHONON_ENERGY_EV * E_CHARGE
    tau_lo_s = TSAI_LO_PHONON_LIFETIME_PS * 1e-12
    if tau_lo_s <= 0:
        return np.nan, np.nan, np.nan

    tau_q_s, q_s_m1 = _tau_c_lo_q_seconds(
        q_m1=q,
        mu_e_ev=mu_e_ev,
        temperature_k=temperature_k,
    )
    n_tc = _phonon_occupation(lo_phonon_energy_j, temperature_k)
    n_tl = _phonon_occupation(lo_phonon_energy_j, lattice_temperature_k)
    if (not np.isfinite(n_tc)) or (not np.isfinite(n_tl)):
        return np.nan, np.nan, np.nan

    n_delta = n_tc - n_tl
    denominator = tau_q_s + tau_lo_s
    integrand = np.where(
        np.isfinite(denominator) & (denominator > 0),
        (q**2) * lo_phonon_energy_j * n_delta / denominator,
        0.0,
    )
    du_dt_intra_w_m3 = -float((1.0 / (2.0 * np.pi**2)) * np.trapezoid(integrand, q))
    du_dt_intra_w_cm3 = du_dt_intra_w_m3 / 1e6

    # Energy-relaxation time (Eq. 48 definition) over the same q-domain used numerically.
    u_delta_integrand = (q**2) * lo_phonon_energy_j * n_delta
    u_delta_j_m3 = float((1.0 / (2.0 * np.pi**2)) * np.trapezoid(u_delta_integrand, q))
    if np.isfinite(du_dt_intra_w_m3) and np.isfinite(u_delta_j_m3) and (du_dt_intra_w_m3 < 0):
        tau_energy_s = u_delta_j_m3 / (-du_dt_intra_w_m3)
    else:
        tau_energy_s = np.nan

    return float(du_dt_intra_w_cm3), float(q_s_m1 / 100.0), float(tau_energy_s)


def _build_mu_grid_ev(experimental_mu_e_ev: np.ndarray) -> np.ndarray:
    if np.isfinite(TSAI_MU_E_GRID_MIN_EV):
        mu_min = float(TSAI_MU_E_GRID_MIN_EV)
    else:
        mu_min = float(np.nanmin(experimental_mu_e_ev) - TSAI_MU_E_GRID_MARGIN_EV)

    if np.isfinite(TSAI_MU_E_GRID_MAX_EV):
        mu_max = float(TSAI_MU_E_GRID_MAX_EV)
    else:
        mu_max = float(np.nanmax(experimental_mu_e_ev) + TSAI_MU_E_GRID_MARGIN_EV)

    if mu_max <= mu_min:
        mu_min, mu_max = mu_min - 0.05, mu_min + 0.05

    return np.linspace(mu_min, mu_max, max(3, int(TSAI_MU_E_GRID_POINTS)), dtype=float)


def _compute_forward_grid(
    mu_grid_ev: np.ndarray,
    temperature_grid_k: np.ndarray,
    q_grid_m1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_mu = mu_grid_ev.size
    n_t = temperature_grid_k.size
    du_dt_grid_w_cm3 = np.full((n_mu, n_t), np.nan, dtype=float)
    qs_grid_cm1 = np.full((n_mu, n_t), np.nan, dtype=float)
    tau_energy_grid_s = np.full((n_mu, n_t), np.nan, dtype=float)

    for i, mu_e_ev in enumerate(mu_grid_ev):
        for j, temperature_k in enumerate(temperature_grid_k):
            du_dt_w_cm3, qs_cm1, tau_energy_s = compute_du_dt_intra_electron_w_cm3(
                mu_e_ev=mu_e_ev,
                temperature_k=temperature_k,
                q_grid_m1=q_grid_m1,
                lattice_temperature_k=TSAI_LATTICE_TEMPERATURE_K,
            )
            du_dt_grid_w_cm3[i, j] = du_dt_w_cm3
            qs_grid_cm1[i, j] = qs_cm1
            tau_energy_grid_s[i, j] = tau_energy_s

    return du_dt_grid_w_cm3, qs_grid_cm1, tau_energy_grid_s


def _forward_grid_to_dataframe(
    mu_grid_ev: np.ndarray,
    temperature_grid_k: np.ndarray,
    du_dt_grid_w_cm3: np.ndarray,
    qs_grid_cm1: np.ndarray,
    tau_energy_grid_s: np.ndarray,
) -> pd.DataFrame:
    mu_mesh, t_mesh = np.meshgrid(mu_grid_ev, temperature_grid_k, indexing="ij")
    p_th_model_w_cm3 = np.maximum(-du_dt_grid_w_cm3, 0.0)
    return pd.DataFrame(
        {
            "mu_e_ev": mu_mesh.ravel(),
            "temperature_k": t_mesh.ravel(),
            "du_dt_intra_w_cm3": du_dt_grid_w_cm3.ravel(),
            "p_th_model_w_cm3": p_th_model_w_cm3.ravel(),
            "q_s_cm1": qs_grid_cm1.ravel(),
            "tau_c_lo_energy_s": tau_energy_grid_s.ravel(),
        }
    )


def _build_inverse_grid(
    mu_grid_ev: np.ndarray,
    temperature_grid_k: np.ndarray,
    p_th_grid_w_cm3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid_pth = p_th_grid_w_cm3[np.isfinite(p_th_grid_w_cm3) & (p_th_grid_w_cm3 > 0)]
    if valid_pth.size < 3:
        raise ValueError("Insufficient valid Tsai forward-grid values to build inverse map.")

    p_th_axis_w_cm3 = np.geomspace(
        float(np.min(valid_pth)),
        float(np.max(valid_pth)),
        max(24, int(TSAI_PTH_INVERSE_POINTS)),
    )
    temperature_inverse = np.full(
        (mu_grid_ev.size, p_th_axis_w_cm3.size),
        np.nan,
        dtype=float,
    )
    log_axis = np.log10(p_th_axis_w_cm3)

    for i in range(mu_grid_ev.size):
        row_pth = p_th_grid_w_cm3[i, :]
        valid = np.isfinite(row_pth) & (row_pth > 0) & np.isfinite(temperature_grid_k)
        if np.count_nonzero(valid) < 2:
            continue

        pth_vals = row_pth[valid]
        t_vals = temperature_grid_k[valid]
        order = np.argsort(pth_vals)
        pth_sorted = pth_vals[order]
        t_sorted = t_vals[order]

        pth_unique, idx_unique = np.unique(pth_sorted, return_index=True)
        if pth_unique.size < 2:
            continue
        t_unique = t_sorted[idx_unique]
        temperature_inverse[i, :] = np.interp(
            log_axis,
            np.log10(pth_unique),
            t_unique,
            left=np.nan,
            right=np.nan,
        )

    return p_th_axis_w_cm3, temperature_inverse


def _inverse_grid_to_dataframe(
    mu_grid_ev: np.ndarray,
    p_th_axis_w_cm3: np.ndarray,
    temperature_inverse_k: np.ndarray,
) -> pd.DataFrame:
    mu_mesh, pth_mesh = np.meshgrid(mu_grid_ev, p_th_axis_w_cm3, indexing="ij")
    return pd.DataFrame(
        {
            "mu_e_ev": mu_mesh.ravel(),
            "p_th_w_cm3": pth_mesh.ravel(),
            "temperature_k": temperature_inverse_k.ravel(),
        }
    )


def _build_temperature_predictor(
    mu_grid_ev: np.ndarray,
    p_th_axis_w_cm3: np.ndarray,
    temperature_inverse_k: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    mu_mesh, pth_mesh = np.meshgrid(mu_grid_ev, p_th_axis_w_cm3, indexing="ij")
    valid = (
        np.isfinite(mu_mesh)
        & np.isfinite(pth_mesh)
        & (pth_mesh > 0)
        & np.isfinite(temperature_inverse_k)
    )
    if np.count_nonzero(valid) < 4:
        raise ValueError("Inverse Tsai map has insufficient finite points for interpolation.")

    points = np.column_stack([mu_mesh[valid], np.log10(pth_mesh[valid])])
    values = temperature_inverse_k[valid]
    linear_interp = LinearNDInterpolator(points, values, fill_value=np.nan)
    nearest_interp = NearestNDInterpolator(points, values)

    def predictor(mu_e_ev: np.ndarray, p_th_w_cm3: np.ndarray) -> np.ndarray:
        mu = np.asarray(mu_e_ev, dtype=float)
        pth = np.asarray(p_th_w_cm3, dtype=float)
        out = np.full_like(mu, np.nan, dtype=float)
        valid_local = np.isfinite(mu) & np.isfinite(pth) & (pth > 0)
        if not np.any(valid_local):
            return out

        query = np.column_stack([mu[valid_local], np.log10(pth[valid_local])])
        pred = np.asarray(linear_interp(query), dtype=float)
        missing = ~np.isfinite(pred)
        if np.any(missing):
            pred[missing] = np.asarray(nearest_interp(query[missing]), dtype=float)
        out[valid_local] = pred
        return out

    return predictor


def _compute_experimental_mu_t_samples(
    exp_df: pd.DataFrame,
    q_grid_m1: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for row in exp_df.itertuples(index=False):
        du_dt_w_cm3, qs_cm1, tau_energy_s = compute_du_dt_intra_electron_w_cm3(
            mu_e_ev=float(row.mu_e_ev),
            temperature_k=float(row.temperature_k_exp),
            q_grid_m1=q_grid_m1,
            lattice_temperature_k=TSAI_LATTICE_TEMPERATURE_K,
        )
        records.append(
            {
                "mu_e_ev": float(row.mu_e_ev),
                "qfls_ev": float(row.qfls_ev),
                "temperature_k_exp": float(row.temperature_k_exp),
                "p_th_exp_w_cm3": float(row.p_th_exp_w_cm3),
                "du_dt_intra_w_cm3_at_exp_mu_t": float(du_dt_w_cm3),
                "p_th_model_w_cm3_at_exp_mu_t": float(max(-du_dt_w_cm3, 0.0)),
                "q_s_cm1_at_exp_mu_t": float(qs_cm1),
                "tau_c_lo_energy_s_at_exp_mu_t": float(tau_energy_s),
            }
        )
    return pd.DataFrame.from_records(records)


def run_tsai_temperature_workflow(
    results_df: pd.DataFrame,
    out_dir: Path,
) -> TsaiWorkflowResult | None:
    if not TSAI_ENABLE_SIMULATION:
        return None

    required = {"mu_e_ev", "qfls_ev", "temperature_k", "thermalized_power_w_cm3"}
    missing = sorted(required - set(results_df.columns))
    if missing:
        raise ValueError(
            "Cannot run Tsai simulation. Missing required columns in results_df: "
            + ", ".join(missing)
        )

    exp_df = results_df.copy()
    exp_df = exp_df.rename(
        columns={
            "temperature_k": "temperature_k_exp",
            "thermalized_power_w_cm3": "p_th_exp_w_cm3",
        }
    )
    exp_df = exp_df[
        [
            "spectrum_id",
            "intensity_w_cm2",
            "mu_e_ev",
            "qfls_ev",
            "temperature_k_exp",
            "p_th_exp_w_cm3",
        ]
    ].copy()
    exp_df = exp_df.replace([np.inf, -np.inf], np.nan).dropna()
    exp_df = exp_df[
        (exp_df["temperature_k_exp"] > 0)
        & (exp_df["p_th_exp_w_cm3"] > 0)
    ]
    if exp_df.shape[0] < 3:
        return None

    q_grid_cm1 = np.geomspace(
        max(1.0, float(TSAI_Q_MIN_CM1)),
        max(float(TSAI_Q_MAX_CM1), float(TSAI_Q_MIN_CM1) * 1.01),
        max(50, int(TSAI_Q_POINTS)),
    )
    q_grid_m1 = q_grid_cm1 * 100.0

    mu_grid_ev = _build_mu_grid_ev(exp_df["mu_e_ev"].to_numpy(dtype=float))
    temperature_grid_k = np.linspace(
        float(TSAI_T_GRID_MIN_K),
        max(float(TSAI_T_GRID_MAX_K), float(TSAI_T_GRID_MIN_K) + 1.0),
        max(6, int(TSAI_T_GRID_POINTS)),
        dtype=float,
    )

    du_dt_grid_w_cm3, qs_grid_cm1, tau_energy_grid_s = _compute_forward_grid(
        mu_grid_ev=mu_grid_ev,
        temperature_grid_k=temperature_grid_k,
        q_grid_m1=q_grid_m1,
    )
    p_th_grid_w_cm3 = np.maximum(-du_dt_grid_w_cm3, 0.0)

    forward_table_df = _forward_grid_to_dataframe(
        mu_grid_ev=mu_grid_ev,
        temperature_grid_k=temperature_grid_k,
        du_dt_grid_w_cm3=du_dt_grid_w_cm3,
        qs_grid_cm1=qs_grid_cm1,
        tau_energy_grid_s=tau_energy_grid_s,
    )
    forward_table_df.to_csv(out_dir / "tsai_forward_muT_to_pth.csv", index=False)

    samples_df = _compute_experimental_mu_t_samples(exp_df=exp_df, q_grid_m1=q_grid_m1)
    samples_df.to_csv(out_dir / "tsai_du_dt_samples_at_experimental_muT.csv", index=False)

    p_th_axis_w_cm3, temperature_inverse_k = _build_inverse_grid(
        mu_grid_ev=mu_grid_ev,
        temperature_grid_k=temperature_grid_k,
        p_th_grid_w_cm3=p_th_grid_w_cm3,
    )
    inverse_table_df = _inverse_grid_to_dataframe(
        mu_grid_ev=mu_grid_ev,
        p_th_axis_w_cm3=p_th_axis_w_cm3,
        temperature_inverse_k=temperature_inverse_k,
    )
    inverse_table_df.to_csv(out_dir / "tsai_inverse_pth_mu_to_temperature.csv", index=False)

    predictor = _build_temperature_predictor(
        mu_grid_ev=mu_grid_ev,
        p_th_axis_w_cm3=p_th_axis_w_cm3,
        temperature_inverse_k=temperature_inverse_k,
    )
    exp_mu = exp_df["mu_e_ev"].to_numpy(dtype=float)
    exp_p_th = exp_df["p_th_exp_w_cm3"].to_numpy(dtype=float)
    temperature_sim_k = predictor(exp_mu, exp_p_th)

    comparison_df = exp_df.copy()
    comparison_df["temperature_sim_k"] = temperature_sim_k
    comparison_df["temperature_error_k"] = (
        comparison_df["temperature_sim_k"] - comparison_df["temperature_k_exp"]
    )
    comparison_df["temperature_abs_error_k"] = np.abs(comparison_df["temperature_error_k"])
    comparison_df["temperature_error_pct"] = (
        100.0
        * comparison_df["temperature_error_k"]
        / comparison_df["temperature_k_exp"]
    )
    comparison_df.to_csv(out_dir / "tsai_temperature_comparison.csv", index=False)

    return TsaiWorkflowResult(
        forward_table_df=forward_table_df,
        inverse_table_df=inverse_table_df,
        experimental_prediction_df=comparison_df,
        mu_grid_ev=mu_grid_ev,
        temperature_grid_k=temperature_grid_k,
        p_th_grid_w_cm3=p_th_grid_w_cm3,
    )
