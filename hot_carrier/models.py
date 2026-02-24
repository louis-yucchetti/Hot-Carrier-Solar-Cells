from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .config import C, E_CHARGE, H, K_B


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
    values: dict[str, float | str | int] = {
        field.name: np.nan for field in fields(FitResult)
    }
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


def _combine_uncertainties(*errors: float) -> float:
    finite_errors = np.array(
        [float(abs(err)) for err in errors if np.isfinite(err)], dtype=float
    )
    if finite_errors.size == 0:
        return np.nan
    return float(np.sqrt(np.sum(finite_errors**2)))
