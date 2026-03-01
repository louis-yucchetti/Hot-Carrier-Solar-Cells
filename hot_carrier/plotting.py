from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
from matplotlib.tri import Triangulation

from .analysis import linearized_signal, _safe_log_yerr, _sanitize_nonnegative
from .config import (
    EG_EV,
    E_CHARGE,
    FIT_RANGE_SCAN_PLOT_WEIGHT_COVERAGE,
    K_B,
    MB_VALIDITY_REL_ERROR_LIMIT,
    SAVE_DPI,
)
from .models import FitResult

if TYPE_CHECKING:
    from .tsai_model import TsaiWorkflowResult


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


def plot_mb_validity_limit(
    curves_df: pd.DataFrame,
    limits_df: pd.DataFrame,
    outpath: Path,
    rel_error_limit: float = MB_VALIDITY_REL_ERROR_LIMIT,
) -> None:
    required_curve_cols = {
        "temperature_k",
        "reduced_qfls",
        "ln_integral_ipc_be",
        "ln_integral_ipc_mb",
        "mb_relative_error",
    }
    if curves_df is None or curves_df.empty:
        return
    if not required_curve_cols.issubset(curves_df.columns):
        return

    df = curves_df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["temperature_k", "reduced_qfls", "ln_integral_ipc_be", "ln_integral_ipc_mb"]
    )
    if df.shape[0] < 6:
        return

    temperatures_k = np.sort(df["temperature_k"].unique())
    if temperatures_k.size == 0:
        return

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(8.7, 6.9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.8, 1.6], "hspace": 0.08},
    )
    colors = cm.viridis(np.linspace(0.15, 0.90, temperatures_k.size))

    for color, temperature_k in zip(colors, temperatures_k, strict=True):
        row = (
            df[df["temperature_k"] == temperature_k]
            .sort_values("reduced_qfls")
            .reset_index(drop=True)
        )
        x = row["reduced_qfls"].to_numpy(dtype=float)
        y_be = row["ln_integral_ipc_be"].to_numpy(dtype=float)
        y_mb = row["ln_integral_ipc_mb"].to_numpy(dtype=float)
        rel = row["mb_relative_error"].to_numpy(dtype=float)

        ax0.plot(
            x,
            y_be,
            color=color,
            lw=1.8,
            label=rf"BE, $T$={temperature_k:.0f} K",
        )
        ax0.plot(
            x,
            y_mb,
            color=color,
            lw=1.2,
            ls="--",
            alpha=0.92,
        )
        ax1.plot(
            x,
            rel,
            color=color,
            lw=1.5,
            label=rf"$T$={temperature_k:.0f} K",
        )

        if (
            limits_df is not None
            and (not limits_df.empty)
            and {"temperature_k", "x_limit"}.issubset(limits_df.columns)
        ):
            limit_row = limits_df[np.isclose(limits_df["temperature_k"], temperature_k)]
            if not limit_row.empty:
                x_limit = float(limit_row["x_limit"].iloc[0])
                if np.isfinite(x_limit):
                    y_limit = np.interp(x_limit, x, y_be, left=np.nan, right=np.nan)
                    rel_limit = np.interp(x_limit, x, rel, left=np.nan, right=np.nan)
                    if np.isfinite(y_limit):
                        ax0.scatter(
                            [x_limit],
                            [y_limit],
                            s=34,
                            color=color,
                            edgecolors="black",
                            linewidths=0.5,
                            zorder=4,
                        )
                    if np.isfinite(rel_limit):
                        ax1.scatter(
                            [x_limit],
                            [rel_limit],
                            s=30,
                            color=color,
                            edgecolors="black",
                            linewidths=0.5,
                            zorder=4,
                        )

    conservative_x = np.nan
    if (
        limits_df is not None
        and (not limits_df.empty)
        and ("x_limit_conservative" in limits_df.columns)
    ):
        conservative_x = float(limits_df["x_limit_conservative"].iloc[0])
    if not np.isfinite(conservative_x) and (
        limits_df is not None
        and (not limits_df.empty)
        and ("x_limit" in limits_df.columns)
    ):
        finite_limits = limits_df["x_limit"].to_numpy(dtype=float)
        finite_limits = finite_limits[np.isfinite(finite_limits)]
        if finite_limits.size > 0:
            conservative_x = float(np.min(finite_limits))

    x_all = df["reduced_qfls"].to_numpy(dtype=float)
    x_left = float(np.nanmin(x_all))
    x_right = float(np.nanmax(x_all))
    if np.isfinite(conservative_x) and (x_right > conservative_x):
        for ax in (ax0, ax1):
            ax.axvspan(
                conservative_x,
                x_right,
                facecolor="#ffebee",
                alpha=0.45,
                zorder=0,
            )
            ax.axvline(conservative_x, color="#b71c1c", lw=1.0, ls=":", alpha=0.9)
        ax0.text(
            0.03,
            0.05,
            rf"Non-MB side starts near $x^* \approx {conservative_x:.2f}$",
            transform=ax0.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.8,
            bbox={
                "facecolor": "white",
                "edgecolor": "0.35",
                "boxstyle": "square,pad=0.2",
                "alpha": 0.92,
            },
        )

    style_axes(ax0)
    ax0.set_ylabel(r"$\ln\!\left(\int_{E_g}^{\infty} I_{\mathrm{PC}}(E)\,\mathrm{d}E\right)$")
    ax0.set_title(
        "Maxwell-Boltzmann validity limit from integrated generalized Planck law"
    )
    top_handles, top_labels = ax0.get_legend_handles_labels()
    top_handles.append(Line2D([0], [0], color="0.25", lw=1.2, ls="--"))
    top_labels.append("MB affine reference")
    ax0.legend(top_handles, top_labels, loc="best", fontsize=8.6)

    ax1.axhline(
        rel_error_limit,
        color="#b71c1c",
        lw=1.2,
        ls="--",
        label=rf"MB error threshold = {100.0 * rel_error_limit:.1f}%",
    )
    style_axes(ax1, logy=True)
    ax1.set_xlabel(r"Reduced QFLS, $(\Delta\mu - E_g)/(k_B T)$")
    ax1.set_ylabel(r"$\Phi_{\mathrm{BE}}/\Phi_{\mathrm{MB}} - 1$")
    ax1.set_title(r"Deviation from MB")
    ax1.legend(loc="best", fontsize=8.6)
    ax1.set_xlim(x_left, x_right)

    if limits_df is not None and (not limits_df.empty):
        lines: list[str] = []
        for row in limits_df.itertuples(index=False):
            if hasattr(row, "x_limit") and np.isfinite(row.x_limit):
                lines.append(f"T={row.temperature_k:.0f} K: x*={row.x_limit:.2f}")
        if lines:
            ax1.text(
                0.03,
                0.04,
                "\n".join(lines),
                transform=ax1.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.5,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "0.35",
                    "boxstyle": "square,pad=0.2",
                    "alpha": 0.92,
                },
            )

    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.09, top=0.95, hspace=0.09)
    save_figure(fig, outpath)
    plt.close(fig)


def plot_tsai_temperature_comparison(
    tsai_result: "TsaiWorkflowResult",
    outpath: Path,
) -> None:
    axis_name = tsai_result.primary_axis_name
    axis_label = (
        r"QFLS, $\Delta\mu$ (eV)"
        if axis_name == "delta_mu_ev"
        else r"Electron chemical potential, $\mu_e$ (eV)"
    )
    inverse_df = tsai_result.inverse_table_df.copy()
    inverse_df = inverse_df.replace([np.inf, -np.inf], np.nan).dropna()
    inverse_df = inverse_df[
        (inverse_df["p_th_w_cm3"] > 0)
        & (inverse_df["temperature_k"] > 0)
    ]
    if axis_name not in inverse_df.columns:
        return
    exp_df = tsai_result.experimental_prediction_df.copy()
    required_exp = [axis_name, "p_th_exp_w_cm3", "temperature_k_exp", "temperature_sim_k"]
    exp_df = exp_df.replace([np.inf, -np.inf], np.nan).dropna(subset=required_exp)
    exp_df = exp_df[
        (exp_df["p_th_exp_w_cm3"] > 0)
        & (exp_df["temperature_k_exp"] > 0)
        & (exp_df["temperature_sim_k"] > 0)
    ]
    if (inverse_df.shape[0] < 4) or (exp_df.shape[0] < 2):
        return

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.6, 4.9))

    tri = Triangulation(
        inverse_df[axis_name].to_numpy(dtype=float),
        np.log10(inverse_df["p_th_w_cm3"].to_numpy(dtype=float)),
    )
    temp_vals = inverse_df["temperature_k"].to_numpy(dtype=float)
    level_min = float(np.nanmin(temp_vals))
    level_max = float(np.nanmax(temp_vals))
    if level_max <= level_min:
        level_max = level_min + 1.0
    levels = np.linspace(level_min, level_max, 22)
    contour = ax0.tricontourf(
        tri,
        temp_vals,
        levels=levels,
        cmap="viridis",
    )
    ax0.scatter(
        exp_df[axis_name],
        exp_df["p_th_exp_w_cm3"],
        s=54,
        facecolors="none",
        edgecolors="white",
        linewidths=1.0,
        zorder=3,
        label="Experimental (mu_e, P_th)",
    )
    ax0.scatter(
        exp_df[axis_name],
        exp_df["p_th_exp_w_cm3"],
        s=16,
        c="k",
        alpha=0.75,
        zorder=4,
    )
    style_axes(ax0, logy=True)
    ax0.set_xlabel(axis_label)
    ax0.set_ylabel(r"$P_{\mathrm{th}}$ (W cm$^{-3}$)")
    ax0.set_title("Simulated inverse map with experimental points")
    ax0.legend(loc="best", fontsize=8.5)
    cbar0 = fig.colorbar(contour, ax=ax0, pad=0.02, fraction=0.055)
    cbar0.set_label("Simulated temperature (K)")

    t_exp = exp_df["temperature_k_exp"].to_numpy(dtype=float)
    t_sim = exp_df["temperature_sim_k"].to_numpy(dtype=float)
    primary_for_color = exp_df[axis_name].to_numpy(dtype=float)
    ax1.scatter(
        t_exp,
        t_sim,
        c=primary_for_color,
        cmap="cividis",
        s=56,
        edgecolors="white",
        linewidths=0.55,
        zorder=3,
    )
    lo = float(min(np.nanmin(t_exp), np.nanmin(t_sim)))
    hi = float(max(np.nanmax(t_exp), np.nanmax(t_sim)))
    parity = np.linspace(lo * 0.95, hi * 1.05, 180)
    ax1.plot(parity, parity, "--", color="0.2", lw=1.1, label="1:1 line")
    style_axes(ax1)
    ax1.set_xlabel(r"Experimental temperature, $T_{\mathrm{exp}}$ (K)")
    ax1.set_ylabel(r"Simulated temperature, $T_{\mathrm{sim}}$ (K)")
    ax1.set_title(r"Pointwise $T_{\mathrm{sim}}$ vs $T_{\mathrm{exp}}$")
    ax1.legend(loc="best", fontsize=8.5)
    mae_k = float(np.nanmean(np.abs(t_sim - t_exp)))
    bias_k = float(np.nanmean(t_sim - t_exp))
    ax1.text(
        0.03,
        0.97,
        f"MAE = {mae_k:.2f} K\nBias = {bias_k:.2f} K",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        bbox={
            "facecolor": "white",
            "edgecolor": "0.35",
            "boxstyle": "square,pad=0.2",
            "alpha": 0.92,
        },
    )

    fig.suptitle("Tsai-model temperature inversion against experiment", y=1.02)
    fig.tight_layout(pad=0.7)
    save_figure(fig, outpath)
    plt.close(fig)


def plot_tsai_temperature_rise_vs_pth_density(
    tsai_result: "TsaiWorkflowResult",
    outpath: Path,
) -> None:
    df = tsai_result.experimental_prediction_df.copy()
    needed = [
        "p_th_exp_w_cm3",
        "temperature_rise_exp_k",
        "temperature_rise_sim_k",
        "carrier_density_exp_cm3",
        "carrier_density_sim_cm3",
    ]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
    if df.shape[0] < 2:
        return

    pth = df["p_th_exp_w_cm3"].to_numpy(dtype=float)
    dT_exp = df["temperature_rise_exp_k"].to_numpy(dtype=float)
    dT_sim = df["temperature_rise_sim_k"].to_numpy(dtype=float)
    n_exp = df["carrier_density_exp_cm3"].to_numpy(dtype=float)
    n_sim = df["carrier_density_sim_cm3"].to_numpy(dtype=float)

    valid = (
        np.isfinite(pth)
        & (pth > 0)
        & np.isfinite(dT_exp)
        & np.isfinite(dT_sim)
        & np.isfinite(n_exp)
        & np.isfinite(n_sim)
        & (n_exp > 0)
        & (n_sim > 0)
    )
    if np.count_nonzero(valid) < 2:
        return

    pth = pth[valid]
    dT_exp = dT_exp[valid]
    dT_sim = dT_sim[valid]
    n_exp = n_exp[valid]
    n_sim = n_sim[valid]

    n_all = np.concatenate([n_exp, n_sim])
    n_norm = LogNorm(vmin=float(np.min(n_all)), vmax=float(np.max(n_all)))
    cmap = cm.viridis

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(8.6, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.5], "hspace": 0.08},
    )
    for x, y0, y1 in zip(pth, dT_exp, dT_sim, strict=True):
        ax0.plot([x, x], [y0, y1], color="0.65", lw=0.8, alpha=0.35, zorder=1)

    ax0.scatter(
        pth,
        dT_exp,
        c=n_exp,
        cmap=cmap,
        norm=n_norm,
        s=58,
        marker="o",
        edgecolors="white",
        linewidths=0.6,
        label="Experimental",
        zorder=3,
    )
    ax0.scatter(
        pth,
        dT_sim,
        c=n_sim,
        cmap=cmap,
        norm=n_norm,
        s=62,
        marker="^",
        edgecolors="black",
        linewidths=0.5,
        label="Tsai-simulated",
        zorder=4,
    )

    style_axes(ax0, logx=True)
    ax0.set_ylabel(r"Carrier temperature rise, $T - T_L$ (K)")
    ax0.set_title(r"Figure of merit: $T-T_L$ vs $P_{\mathrm{th}}$ colored by carrier density")
    ax0.legend(loc="best", fontsize=9)

    delta_t = dT_sim - dT_exp
    ax1.axhline(0.0, color="0.25", lw=1.0, ls="--", zorder=1)
    ax1.scatter(
        pth,
        delta_t,
        c=n_exp,
        cmap=cmap,
        norm=n_norm,
        s=48,
        marker="o",
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    style_axes(ax1, logx=True)
    ax1.set_xlabel(r"$P_{\mathrm{th}}$ (W cm$^{-3}$)")
    ax1.set_ylabel(r"$\Delta T_{\mathrm{sim-exp}}$ (K)")
    ax1.set_title(r"Residual diagnostic")

    sm = cm.ScalarMappable(norm=n_norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=[ax0, ax1], pad=0.02, fraction=0.045)
    cbar.set_label(r"Carrier density, $n$ (cm$^{-3}$)")

    mae_rise = float(np.nanmean(np.abs(delta_t)))
    bias_rise = float(np.nanmean(delta_t))
    ax1.text(
        0.03,
        0.96,
        f"MAE(ΔT) = {mae_rise:.2f} K\nBias(ΔT) = {bias_rise:.2f} K",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
        bbox={
            "facecolor": "white",
            "edgecolor": "0.35",
            "boxstyle": "square,pad=0.2",
            "alpha": 0.92,
        },
    )

    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.10, top=0.95, hspace=0.10)
    save_figure(fig, outpath)
    plt.close(fig)
