from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (
    build_mb_validity_scan,
    compute_power_balance_table,
    fit_single_spectrum,
    load_spectra,
)
from .config import (
    ABSORPTIVITY_AT_LASER,
    ABSORPTIVITY_AT_LASER_SIGMA,
    ACTIVE_LAYER_THICKNESS_NM,
    ASSUMED_A0,
    A0_SIGMA,
    A0_UNCERTAINTY_MODEL,
    AUTO_SELECT_FIT_WINDOW,
    EG_EV,
    ESTIMATE_FIT_RANGE_UNCERTAINTY,
    EXCITATION_INTENSITY_W_CM2,
    FILENAME,
    FIT_ENERGY_MAX_EV,
    FIT_ENERGY_MIN_EV,
    FIT_RANGE_SCAN_MIN_POINTS,
    FIT_RANGE_SCAN_MIN_R2,
    FIT_RANGE_SCAN_PLOT_WEIGHT_COVERAGE,
    LASER_WAVELENGTH_NM,
    MB_VALIDITY_ENABLE,
    MB_VALIDITY_REFERENCE_TEMPERATURES_K,
    MB_VALIDITY_REL_ERROR_LIMIT,
    H,
    C,
    PLQY_ETA,
    PLQY_ETA_SIGMA,
    PLQY_RESULTS_CSV,
    TSAI_ENABLE_SIMULATION,
    TSAI_MODEL_TABLE_CSV,
    WINDOW_PEAK_OFFSET_EV,
    WINDOW_SEARCH_MAX_EV,
    WINDOW_SEARCH_MIN_EV,
)
from .models import FitResult
from .plotting import (
    load_tsai_model_table,
    plot_pth_nt_comparison,
    plot_raw_spectra,
    plot_single_fit,
    plot_summary,
    plot_mb_validity_limit,
    plot_thermalized_power_diagnostics,
    plot_tsai_temperature_rise_vs_pth_density,
    setup_plot_style,
)
from .tsai_model import TsaiWorkflowResult, run_tsai_temperature_workflow


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


def _select_mb_validity_temperatures(results_df: pd.DataFrame) -> np.ndarray:
    if "temperature_k" not in results_df.columns:
        return np.asarray(MB_VALIDITY_REFERENCE_TEMPERATURES_K, dtype=float)

    temperatures = results_df["temperature_k"].to_numpy(dtype=float)
    temperatures = temperatures[np.isfinite(temperatures) & (temperatures > 0)]
    if temperatures.size == 0:
        return np.asarray(MB_VALIDITY_REFERENCE_TEMPERATURES_K, dtype=float)

    if temperatures.size >= 3:
        selected = np.quantile(temperatures, [0.1, 0.5, 0.9])
    else:
        selected = temperatures
    selected = np.unique(np.round(selected.astype(float), 2))
    if selected.size == 0:
        selected = np.asarray(MB_VALIDITY_REFERENCE_TEMPERATURES_K, dtype=float)
    return selected


def _normalize_column_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _find_first_present_column(
    columns: list[str],
    candidates: tuple[str, ...],
) -> str | None:
    normalized = {_normalize_column_name(col): col for col in columns}
    for candidate in candidates:
        key = _normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def _to_fraction(values: np.ndarray, column_name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    col_norm = _normalize_column_name(column_name)
    if ("percent" in col_norm) or ("plqy" in col_norm and "%" in column_name):
        return arr / 100.0
    if np.nanmax(finite) > 1.0:
        return arr / 100.0
    return arr


def _resolve_plqy_profiles(
    root: Path,
    intensities_w_cm2: np.ndarray,
    laser_wavelength_nm: float,
    absorptivity_at_laser: float,
    default_eta: float,
    default_eta_sigma: float,
    plqy_csv: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    n_points = intensities_w_cm2.size
    eta_default = np.full(n_points, float(default_eta), dtype=float)
    sigma_default = np.full(n_points, float(default_eta_sigma), dtype=float)

    csv_path_str = str(plqy_csv).strip()
    if not csv_path_str:
        return eta_default, sigma_default, "constant"

    csv_path = Path(csv_path_str)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    if not csv_path.is_file():
        print(f"Warning: PLQY file not found, using constant PLQY: {csv_path}")
        return eta_default, sigma_default, "constant"

    table = pd.read_csv(csv_path)
    if table.empty:
        print(f"Warning: PLQY file is empty, using constant PLQY: {csv_path}")
        return eta_default, sigma_default, "constant"

    columns = list(table.columns.astype(str))
    plqy_col = _find_first_present_column(columns, ("PLQY (%)", "PLQY", "eta"))
    err_col = _find_first_present_column(
        columns,
        ("err_PLQY (%)", "PLQY_err (%)", "err_PLQY", "PLQY_err", "eta_err"),
    )
    phiabs_col = _find_first_present_column(
        columns,
        ("phiabs (photons/s)", "phiabs", "phi_abs_photons_s"),
    )
    intensity_col = _find_first_present_column(
        columns,
        ("intensity_w_cm2", "excitation_intensity_w_cm2", "i_exc_w_cm2"),
    )

    if plqy_col is None:
        print(
            f"Warning: PLQY column not found in {csv_path}; using constant PLQY."
        )
        return eta_default, sigma_default, "constant"

    plqy_raw = pd.to_numeric(table[plqy_col], errors="coerce").to_numpy(dtype=float)
    eta_table = _to_fraction(plqy_raw, plqy_col)
    if err_col is None:
        sigma_table = np.full_like(eta_table, float(default_eta_sigma), dtype=float)
    else:
        err_raw = pd.to_numeric(table[err_col], errors="coerce").to_numpy(dtype=float)
        sigma_table = _to_fraction(err_raw, err_col)

    eta_out = eta_default.copy()
    sigma_out = sigma_default.copy()

    if phiabs_col is not None:
        phiabs_table = pd.to_numeric(table[phiabs_col], errors="coerce").to_numpy(dtype=float)
        e_laser_j = (H * C) / (laser_wavelength_nm * 1e-9)
        phiabs_target = (absorptivity_at_laser * intensities_w_cm2) / e_laser_j

        valid = (
            np.isfinite(phiabs_table)
            & (phiabs_table > 0)
            & np.isfinite(eta_table)
            & np.isfinite(sigma_table)
        )
        if np.count_nonzero(valid) >= 2:
            x = np.log10(phiabs_table[valid])
            y_eta = eta_table[valid]
            y_sigma = sigma_table[valid]
            order = np.argsort(x)
            x_sorted = x[order]
            eta_sorted = y_eta[order]
            sigma_sorted = y_sigma[order]
            x_unique, unique_idx = np.unique(x_sorted, return_index=True)
            eta_unique = eta_sorted[unique_idx]
            sigma_unique = sigma_sorted[unique_idx]
            x_target = np.log10(phiabs_target)
            in_domain = (x_target >= np.min(x_unique)) & (x_target <= np.max(x_unique))
            min_in_domain = max(2, int(0.30 * n_points))
            if np.count_nonzero(in_domain) >= min_in_domain:
                eta_out = np.interp(x_target, x_unique, eta_unique)
                sigma_out = np.interp(x_target, x_unique, sigma_unique)
                eta_out = np.clip(eta_out, 0.0, 1.0)
                sigma_out = np.clip(sigma_out, 0.0, np.inf)
                return eta_out, sigma_out, f"table:{csv_path.name} (interpolated vs phiabs)"
            print(
                "Warning: PLQY phiabs scale does not overlap experimental phi_abs; "
                "skipping phiabs interpolation."
            )

    if intensity_col is not None:
        intensity_table = pd.to_numeric(table[intensity_col], errors="coerce").to_numpy(dtype=float)
        valid = (
            np.isfinite(intensity_table)
            & (intensity_table > 0)
            & np.isfinite(eta_table)
            & np.isfinite(sigma_table)
        )
        if np.count_nonzero(valid) >= 2:
            x = np.log10(intensity_table[valid])
            y_eta = eta_table[valid]
            y_sigma = sigma_table[valid]
            order = np.argsort(x)
            x_sorted = x[order]
            eta_sorted = y_eta[order]
            sigma_sorted = y_sigma[order]
            x_unique, unique_idx = np.unique(x_sorted, return_index=True)
            eta_unique = eta_sorted[unique_idx]
            sigma_unique = sigma_sorted[unique_idx]

            eta_out = np.interp(np.log10(intensities_w_cm2), x_unique, eta_unique)
            sigma_out = np.interp(np.log10(intensities_w_cm2), x_unique, sigma_unique)
            eta_out = np.clip(eta_out, 0.0, 1.0)
            sigma_out = np.clip(sigma_out, 0.0, np.inf)
            return eta_out, sigma_out, f"table:{csv_path.name} (interpolated vs intensity)"

    valid_rows = np.isfinite(eta_table) & np.isfinite(sigma_table)
    if np.count_nonzero(valid_rows) == n_points:
        eta_out = np.clip(eta_table, 0.0, 1.0)
        sigma_out = np.clip(sigma_table, 0.0, np.inf)
        return eta_out, sigma_out, f"table:{csv_path.name} (row-aligned)"

    print(
        f"Warning: Could not map PLQY table to excitation points; using constant PLQY."
    )
    return eta_default, sigma_default, "constant"


def _print_run_summary(
    out_dir: Path,
    fit_dir: Path,
    comparison_df: pd.DataFrame | None,
    tsai_model_df: pd.DataFrame | None,
    tsai_simulation_result: TsaiWorkflowResult | None,
    mb_validity_curves_df: pd.DataFrame | None,
    mb_validity_limits_df: pd.DataFrame | None,
    plqy_source: str,
    plqy_eta_used: np.ndarray,
    plqy_eta_sigma_used: np.ndarray,
) -> None:
    print("Done.")
    print(f"Raw spectra plot: {out_dir / 'all_spectra_logscale.png'}")
    print(f"Spectrum fits:    {fit_dir}")
    print(f"Results table:    {out_dir / 'fit_results.csv'}")
    print(f"Summary figure:   {out_dir / 'parameters_vs_intensity.png'}")
    print(f"Power figure:     {out_dir / 'thermalized_power_diagnostics.png'}")
    if mb_validity_curves_df is not None and mb_validity_limits_df is not None:
        print(f"MB limit figure:  {out_dir / 'mb_validity_limit.png'}")
        print(f"MB curves CSV:    {out_dir / 'mb_validity_scan.csv'}")
        print(f"MB limits CSV:    {out_dir / 'mb_validity_limits.csv'}")
    else:
        print("MB limit figure:  skipped")
    if tsai_model_df is not None:
        print(f"Pth(n,T) figure:  {out_dir / 'pth_nT_comparison.png'}")
    else:
        print("Pth(n,T) figure:  skipped (no external Tsai model table)")
    if tsai_simulation_result is not None:
        print(f"Tsai FOM plot:    {out_dir / 'tsai_temperature_rise_vs_pth_density.png'}")
    if comparison_df is not None:
        print(f"Tsai compare CSV: {out_dir / 'pth_experiment_vs_tsai.csv'}")
    if tsai_simulation_result is not None:
        print(f"Tsai forward CSV: {tsai_simulation_result.forward_csv_path}")
        print(f"Tsai inverse CSV: {tsai_simulation_result.inverse_csv_path}")
        print(f"Tsai sample CSV:  {tsai_simulation_result.samples_csv_path}")
        print(f"Tsai T-CSV:       {tsai_simulation_result.comparison_csv_path}")
        print(
            "Tsai model run:   Eq.41 + Eq.48 | "
            f"forward grid points={tsai_simulation_result.forward_table_df.shape[0]}"
        )
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
        rf"PLQY source={plqy_source}, "
        rf"eta=[{np.min(plqy_eta_used):.4f}, {np.max(plqy_eta_used):.4f}], "
        rf"sigma=[{np.min(plqy_eta_sigma_used):.4f}, {np.max(plqy_eta_sigma_used):.4f}], "
        rf"d={ACTIVE_LAYER_THICKNESS_NM:.1f} nm"
    )
    if tsai_model_df is None:
        print("Tsai model table: not provided (set TSAI_MODEL_TABLE_CSV to enable overlay/parity)")
    else:
        print(
            "Tsai model table: "
            f"{TSAI_MODEL_TABLE_CSV} | points={tsai_model_df.shape[0]}"
        )
    if mb_validity_limits_df is not None and (not mb_validity_limits_df.empty):
        finite_limits = mb_validity_limits_df["x_limit"].to_numpy(dtype=float)
        finite_limits = finite_limits[np.isfinite(finite_limits)]
        if finite_limits.size > 0:
            print(
                "MB validity:     "
                f"x*≈{np.min(finite_limits):.2f} for "
                f"rel.error>{100.0 * MB_VALIDITY_REL_ERROR_LIMIT:.1f}%"
            )



def main() -> None:
    setup_plot_style()
    _validate_configuration()

    root = Path(__file__).resolve().parent.parent
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

    plqy_eta_profile, plqy_eta_sigma_profile, plqy_source = _resolve_plqy_profiles(
        root=root,
        intensities_w_cm2=EXCITATION_INTENSITY_W_CM2,
        laser_wavelength_nm=LASER_WAVELENGTH_NM,
        absorptivity_at_laser=ABSORPTIVITY_AT_LASER,
        default_eta=PLQY_ETA,
        default_eta_sigma=PLQY_ETA_SIGMA,
        plqy_csv=PLQY_RESULTS_CSV,
    )

    results_df = pd.DataFrame([r.__dict__ for r in all_results])
    results_df = compute_power_balance_table(
        results_df=results_df,
        laser_wavelength_nm=LASER_WAVELENGTH_NM,
        absorptivity_at_laser=ABSORPTIVITY_AT_LASER,
        absorptivity_at_laser_sigma=ABSORPTIVITY_AT_LASER_SIGMA,
        plqy_eta=plqy_eta_profile,
        plqy_eta_sigma=plqy_eta_sigma_profile,
        active_layer_thickness_nm=ACTIVE_LAYER_THICKNESS_NM,
        eg_ev=EG_EV,
    )
    tsai_model_df = load_tsai_model_table(TSAI_MODEL_TABLE_CSV)
    pth_nt_outpath = out_dir / "pth_nT_comparison.png"
    comparison_df: pd.DataFrame | None = None
    if tsai_model_df is not None:
        comparison_df = plot_pth_nt_comparison(
            results_df=results_df,
            outpath=pth_nt_outpath,
            theory_df=tsai_model_df,
        )
    elif pth_nt_outpath.exists():
        pth_nt_outpath.unlink()
    results_df.to_csv(out_dir / "fit_results.csv", index=False)
    plot_summary(results_df, out_dir / "parameters_vs_intensity.png")
    legacy_power_plot = out_dir / "thermalized_power_vs_absorbed.png"
    if legacy_power_plot.exists():
        legacy_power_plot.unlink()
    plot_thermalized_power_diagnostics(results_df, out_dir / "thermalized_power_diagnostics.png")
    mb_validity_curves_df: pd.DataFrame | None = None
    mb_validity_limits_df: pd.DataFrame | None = None
    mb_limit_plot_path = out_dir / "mb_validity_limit.png"
    mb_curve_csv_path = out_dir / "mb_validity_scan.csv"
    mb_limit_csv_path = out_dir / "mb_validity_limits.csv"
    if MB_VALIDITY_ENABLE:
        mb_temperatures_k = _select_mb_validity_temperatures(results_df)
        mb_validity_curves_df, mb_validity_limits_df = build_mb_validity_scan(
            temperatures_k=mb_temperatures_k,
            rel_error_limit=MB_VALIDITY_REL_ERROR_LIMIT,
        )
        if (mb_validity_curves_df is not None) and (not mb_validity_curves_df.empty):
            mb_validity_curves_df.to_csv(mb_curve_csv_path, index=False)
            if (mb_validity_limits_df is not None) and (not mb_validity_limits_df.empty):
                mb_validity_limits_df.to_csv(mb_limit_csv_path, index=False)
            plot_mb_validity_limit(
                curves_df=mb_validity_curves_df,
                limits_df=mb_validity_limits_df
                if mb_validity_limits_df is not None
                else pd.DataFrame(),
                outpath=mb_limit_plot_path,
                rel_error_limit=MB_VALIDITY_REL_ERROR_LIMIT,
            )
        else:
            if mb_limit_plot_path.exists():
                mb_limit_plot_path.unlink()
            if mb_curve_csv_path.exists():
                mb_curve_csv_path.unlink()
            if mb_limit_csv_path.exists():
                mb_limit_csv_path.unlink()
    else:
        if mb_limit_plot_path.exists():
            mb_limit_plot_path.unlink()
        if mb_curve_csv_path.exists():
            mb_curve_csv_path.unlink()
        if mb_limit_csv_path.exists():
            mb_limit_csv_path.unlink()
    legacy_tsai_temp_comparison_plot = out_dir / "tsai_temperature_comparison.png"
    if legacy_tsai_temp_comparison_plot.exists():
        legacy_tsai_temp_comparison_plot.unlink()
    tsai_simulation_result: TsaiWorkflowResult | None = None
    if TSAI_ENABLE_SIMULATION:
        tsai_simulation_result = run_tsai_temperature_workflow(
            results_df=results_df,
            out_dir=out_dir,
        )
        if tsai_simulation_result is not None:
            plot_tsai_temperature_rise_vs_pth_density(
                tsai_result=tsai_simulation_result,
                outpath=out_dir / "tsai_temperature_rise_vs_pth_density.png",
            )
    if comparison_df is not None:
        comparison_df.to_csv(out_dir / "pth_experiment_vs_tsai.csv", index=False)
    _print_run_summary(
        out_dir=out_dir,
        fit_dir=fit_dir,
        comparison_df=comparison_df,
        tsai_model_df=tsai_model_df,
        tsai_simulation_result=tsai_simulation_result,
        mb_validity_curves_df=mb_validity_curves_df,
        mb_validity_limits_df=mb_validity_limits_df,
        plqy_source=plqy_source,
        plqy_eta_used=plqy_eta_profile,
        plqy_eta_sigma_used=plqy_eta_sigma_profile,
    )
