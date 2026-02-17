from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import LogNorm


# ----------------------------- User-tunable settings -----------------------------
FILENAME = "GaAs bulk_PL_avg_circle_4pixs.txt"
FIT_ENERGY_MIN_EV = 1.50
FIT_ENERGY_MAX_EV = 1.72

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


def fit_single_spectrum(
    energy_ev: np.ndarray,
    intensity: np.ndarray,
    spectrum_id: str,
    intensity_w_cm2: float,
    fit_min_ev: float,
    fit_max_ev: float,
    assumed_a0: float,
) -> tuple[FitResult, np.ndarray]:
    valid = np.isfinite(energy_ev) & np.isfinite(intensity) & (intensity > 0)
    in_window = (energy_ev >= fit_min_ev) & (energy_ev <= fit_max_ev)
    fit_mask = valid & in_window

    if np.count_nonzero(fit_mask) < 3:
        result = FitResult(
            spectrum_id=spectrum_id,
            intensity_w_cm2=float(intensity_w_cm2),
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
        )
        return result, np.full_like(intensity, np.nan, dtype=float)

    x_j = energy_ev[fit_mask] * E_CHARGE
    y = linearized_signal(energy_ev[fit_mask], intensity[fit_mask])

    slope, intercept = np.polyfit(x_j, y, deg=1)
    y_fit = slope * x_j + intercept
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    temperature_k = -1.0 / (K_B * slope)
    qfls_effective_j = intercept * K_B * temperature_k
    qfls_effective_ev = qfls_effective_j / E_CHARGE
    qfls_ev = qfls_effective_ev - (K_B * temperature_k / E_CHARGE) * np.log(assumed_a0)

    mu_e_ev, mu_h_ev, n_cm3 = compute_mu_and_density_mb(temperature_k=temperature_k, qfls_ev=qfls_ev)

    energy_j_all = energy_ev * E_CHARGE
    ln_prefactor = np.log(2.0 * energy_j_all**2 / (H**3 * C**2))
    ln_i_model = ln_prefactor - (energy_j_all - qfls_effective_j) / (K_B * temperature_k)
    intensity_model = np.exp(ln_i_model)

    result = FitResult(
        spectrum_id=spectrum_id,
        intensity_w_cm2=float(intensity_w_cm2),
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
    )
    return result, intensity_model


def plot_raw_spectra(
    energy_ev: np.ndarray,
    spectra: np.ndarray,
    intensities_w_cm2: np.ndarray,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    norm = LogNorm(vmin=np.min(intensities_w_cm2), vmax=np.max(intensities_w_cm2))
    cmap = cm.viridis

    for i in range(spectra.shape[1]):
        ax.plot(energy_ev, spectra[:, i], color=cmap(norm(intensities_w_cm2[i])), lw=1.0)

    ax.set_yscale("log")
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("PL intensity (a.u.)")
    ax.set_title("All PL spectra (log scale)")
    ax.grid(alpha=0.25, which="both")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Excitation intensity (W/cm^2)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_single_fit(
    energy_ev: np.ndarray,
    intensity: np.ndarray,
    intensity_model: np.ndarray,
    result: FitResult,
    fit_min_ev: float,
    fit_max_ev: float,
    outpath: Path,
) -> None:
    fit_mask = (energy_ev >= fit_min_ev) & (energy_ev <= fit_max_ev) & (intensity > 0)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.0), sharex=False)
    ax0, ax1 = axes

    ax0.plot(energy_ev, intensity, lw=1.4, label="Experiment")
    ax0.plot(energy_ev, intensity_model, lw=1.2, ls="--", label="High-energy linear fit")
    ax0.axvspan(fit_min_ev, fit_max_ev, color="gray", alpha=0.15, label="Fit window")
    ax0.set_yscale("log")
    ax0.set_xlabel("Photon energy (eV)")
    ax0.set_ylabel("PL intensity (a.u.)")
    ax0.grid(alpha=0.25, which="both")
    ax0.legend(loc="best")
    ax0.set_title(
        f"Spectrum {result.spectrum_id} | Iexc={result.intensity_w_cm2:.3g} W/cm^2\n"
        f"T={result.temperature_k:.1f} K, QFLS={result.qfls_ev:.3f} eV, R^2={result.r2:.5f}"
    )

    y_all = linearized_signal(energy_ev[intensity > 0], intensity[intensity > 0])
    ax1.plot(energy_ev[intensity > 0], y_all, lw=1.1, label="Linearized data")

    x_fit_ev = energy_ev[fit_mask]
    x_fit_j = x_fit_ev * E_CHARGE
    y_line = result.slope * x_fit_j + result.intercept
    ax1.plot(x_fit_ev, y_line, lw=1.6, ls="--", label="Linear regression")
    ax1.axvspan(fit_min_ev, fit_max_ev, color="gray", alpha=0.15)
    ax1.set_xlabel("Photon energy (eV)")
    ax1.set_ylabel("log((h^3 c^2 / 2E^2) I)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_summary(results_df: pd.DataFrame, outpath: Path) -> None:
    x = results_df["intensity_w_cm2"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.0))
    ax00, ax01, ax10, ax11 = axes.ravel()

    ax00.plot(x, results_df["temperature_k"], "o-", lw=1.2, ms=4)
    ax00.set_xscale("log")
    ax00.set_xlabel("Excitation intensity (W/cm^2)")
    ax00.set_ylabel("Temperature (K)")
    ax00.grid(alpha=0.3, which="both")

    ax01.plot(x, results_df["qfls_ev"], "o-", lw=1.2, ms=4, label="QFLS")
    ax01.plot(x, results_df["qfls_effective_ev"], "s--", lw=1.0, ms=3, label="QFLS effective")
    ax01.set_xscale("log")
    ax01.set_xlabel("Excitation intensity (W/cm^2)")
    ax01.set_ylabel("QFLS (eV)")
    ax01.grid(alpha=0.3, which="both")
    ax01.legend(loc="best")

    ax10.plot(x, results_df["mu_e_ev"], "o-", lw=1.2, ms=4, label="mu_e")
    ax10.plot(x, results_df["mu_h_ev"], "o-", lw=1.2, ms=4, label="mu_h")
    ax10.set_xscale("log")
    ax10.set_xlabel("Excitation intensity (W/cm^2)")
    ax10.set_ylabel("Chemical potentials (eV)")
    ax10.grid(alpha=0.3, which="both")
    ax10.legend(loc="best")

    ax11.plot(x, results_df["carrier_density_cm3"], "o-", lw=1.2, ms=4)
    ax11.set_xscale("log")
    ax11.set_yscale("log")
    ax11.set_xlabel("Excitation intensity (W/cm^2)")
    ax11.set_ylabel("Carrier density n (cm^-3)")
    ax11.grid(alpha=0.3, which="both")

    fig.suptitle("Extracted hot-carrier parameters vs excitation intensity", y=0.98)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
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
        fit_min_ev=FIT_ENERGY_MIN_EV,
        fit_max_ev=FIT_ENERGY_MAX_EV,
        assumed_a0=ASSUMED_A0,
    )
    plot_single_fit(
        energy_ev=energy_ev,
        intensity=spectra[:, 0],
        intensity_model=first_model,
        result=first_result,
        fit_min_ev=FIT_ENERGY_MIN_EV,
        fit_max_ev=FIT_ENERGY_MAX_EV,
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
            fit_min_ev=FIT_ENERGY_MIN_EV,
            fit_max_ev=FIT_ENERGY_MAX_EV,
            assumed_a0=ASSUMED_A0,
        )
        all_results.append(result)
        plot_single_fit(
            energy_ev=energy_ev,
            intensity=spectra[:, i],
            intensity_model=intensity_model,
            result=result,
            fit_min_ev=FIT_ENERGY_MIN_EV,
            fit_max_ev=FIT_ENERGY_MAX_EV,
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
    print(
        f"Fit window used:  [{FIT_ENERGY_MIN_EV:.3f}, {FIT_ENERGY_MAX_EV:.3f}] eV | "
        f"ASSUMED_A0={ASSUMED_A0:g}"
    )


if __name__ == "__main__":
    main()
