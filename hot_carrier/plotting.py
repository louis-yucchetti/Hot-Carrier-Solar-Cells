from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
from matplotlib.tri import Triangulation

from .analysis import (
    _safe_log_yerr,
    _safe_ratio,
    _sanitize_nonnegative,
    linearized_signal,
)
from .config import (
    EG_EV,
    E_CHARGE,
    FIT_RANGE_SCAN_PLOT_WEIGHT_COVERAGE,
    K_B,
    MB_VALIDITY_REL_ERROR_LIMIT,
    SAVE_DPI,
    TSAI_LATTICE_TEMPERATURE_K,
    TSAI_LO_PHONON_ENERGY_EV,
)
from .models import FitResult

if TYPE_CHECKING:
    from .tsai_model import TsaiWorkflowResult


IEEE_PAGE_SINGLE_COLUMN_WIDTH_IN = 7.16
PAGE_SHORT_FIG_HEIGHT_IN = 4.1
PAGE_MEDIUM_FIG_HEIGHT_IN = 5.3
PAGE_TALL_FIG_HEIGHT_IN = 6.2
PAGE_GRID_FIG_HEIGHT_IN = 5.8
PAGE_LARGE_GRID_FIG_HEIGHT_IN = 6.4
PAGE_XL_GRID_FIG_HEIGHT_IN = 7.2

AXES_LABEL_FONT_SIZE = 9.4
TITLE_FONT_SIZE = 9.8
MULTIPANEL_TITLE_FONT_SIZE = 9.0
TICK_FONT_SIZE = 8.4
LEGEND_FONT_SIZE = 8.0
ANNOTATION_FONT_SIZE = 8.0
PANEL_LABEL_FONT_SIZE = 8.8
SUPTITLE_FONT_SIZE = 10.2
PRESENTATION_PLOT_DIRNAME = "presentation_plots"


def _page_single_column_figsize(height_in: float) -> tuple[float, float]:
    return (IEEE_PAGE_SINGLE_COLUMN_WIDTH_IN, height_in)


def _style_colorbar(colorbar, label: str) -> None:
    colorbar.set_label(label, fontsize=AXES_LABEL_FONT_SIZE)
    colorbar.ax.tick_params(direction="in", labelsize=TICK_FONT_SIZE)


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.03,
        0.93,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_FONT_SIZE,
        fontweight="semibold",
        ha="left",
        va="top",
    )


def _raise_annotation(text_artist: plt.Text) -> None:
    text_artist.set_zorder(30)
    bbox_patch = text_artist.get_bbox_patch()
    if bbox_patch is not None:
        bbox_patch.set_zorder(29)


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
            "axes.labelsize": AXES_LABEL_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.9,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 4.0,
            "xtick.minor.size": 2.2,
            "ytick.major.size": 4.0,
            "ytick.minor.size": 2.2,
            "legend.frameon": True,
            "legend.framealpha": 0.93,
            "legend.fancybox": False,
            "legend.edgecolor": "0.25",
            "legend.fontsize": LEGEND_FONT_SIZE,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.35,
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
    ax.grid(True, which="major", linewidth=0.55)
    ax.grid(True, which="minor", linewidth=0.28, alpha=0.12)


def save_figure(fig: plt.Figure, outpath: Path) -> None:
    png_outpath = outpath.with_suffix(".png")
    fig.savefig(png_outpath, dpi=SAVE_DPI, bbox_inches="tight")


def _presentation_plot_dir(outpath: Path) -> Path:
    panel_dir = outpath.parent / PRESENTATION_PLOT_DIRNAME / outpath.stem
    panel_dir.mkdir(parents=True, exist_ok=True)
    return panel_dir


def _export_single_panel_figure(
    panel_dir: Path,
    filename: str,
    panel_plotter: Callable[[plt.Figure, plt.Axes], None],
    *,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    subplots_adjust: dict[str, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=figsize or _page_single_column_figsize(PAGE_MEDIUM_FIG_HEIGHT_IN))
    panel_plotter(fig, ax)
    if title:
        ax.set_title(title, pad=8.0)
    fig.subplots_adjust(**(subplots_adjust or {"left": 0.12, "right": 0.93, "bottom": 0.12, "top": 0.90}))
    save_figure(fig, panel_dir / filename)
    plt.close(fig)


def _compute_tsai_q_factor_fit(
    pth_w_cm3: np.ndarray,
    temperature_rise_k: np.ndarray,
) -> tuple[float, float, float] | None:
    pth = np.asarray(pth_w_cm3, dtype=float)
    delta_t = np.asarray(temperature_rise_k, dtype=float)
    temperature_k = TSAI_LATTICE_TEMPERATURE_K + delta_t

    valid = (
        np.isfinite(pth)
        & (pth > 0)
        & np.isfinite(delta_t)
        & (delta_t > 0)
        & np.isfinite(temperature_k)
        & (temperature_k > 0)
    )
    if np.count_nonzero(valid) < 2:
        return None

    x = delta_t[valid]
    if np.allclose(x, x[0]):
        return None

    lo_phonon_energy_j = TSAI_LO_PHONON_ENERGY_EV * E_CHARGE
    y = pth[valid] * np.exp(lo_phonon_energy_j / (K_B * temperature_k[valid]))
    slope, intercept = np.polyfit(x, y, deg=1)
    y_fit = slope * x + intercept
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(intercept), float(r2)


def _build_tsai_q_fit_curve(
    temperature_rise_k: np.ndarray,
    slope: float,
    intercept: float,
    n_points: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    delta_t = np.asarray(temperature_rise_k, dtype=float)
    valid = np.isfinite(delta_t) & (delta_t > 0)
    if np.count_nonzero(valid) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    delta_t_min = float(np.min(delta_t[valid]))
    delta_t_max = float(np.max(delta_t[valid]))
    if delta_t_max <= delta_t_min:
        return np.array([], dtype=float), np.array([], dtype=float)

    delta_t_grid = np.linspace(delta_t_min, delta_t_max, int(n_points), dtype=float)
    temperature_k = TSAI_LATTICE_TEMPERATURE_K + delta_t_grid
    lo_phonon_energy_j = TSAI_LO_PHONON_ENERGY_EV * E_CHARGE
    linearized_pth = slope * delta_t_grid + intercept
    pth_fit = linearized_pth * np.exp(-(lo_phonon_energy_j / (K_B * temperature_k)))

    valid_curve = (
        np.isfinite(pth_fit)
        & (pth_fit > 0)
        & np.isfinite(delta_t_grid)
        & (delta_t_grid > 0)
    )
    return pth_fit[valid_curve], delta_t_grid[valid_curve]


def _build_tsai_q_fit_temperature_vs_intensity_curve(
    intensity_w_cm2: np.ndarray,
    pth_w_cm3: np.ndarray,
    temperature_rise_k: np.ndarray,
    slope: float,
    intercept: float,
) -> tuple[np.ndarray, np.ndarray]:
    intensity = np.asarray(intensity_w_cm2, dtype=float)
    pth_target = np.asarray(pth_w_cm3, dtype=float)
    delta_t_reference = np.asarray(temperature_rise_k, dtype=float)

    fit_pth, fit_dt = _build_tsai_q_fit_curve(
        temperature_rise_k=delta_t_reference,
        slope=slope,
        intercept=intercept,
        n_points=480,
    )
    if fit_pth.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    order_fit = np.argsort(fit_pth)
    pth_sorted = fit_pth[order_fit]
    dt_sorted = fit_dt[order_fit]
    pth_unique, unique_idx = np.unique(pth_sorted, return_index=True)
    dt_unique = dt_sorted[unique_idx]
    if pth_unique.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    valid = (
        np.isfinite(intensity)
        & (intensity > 0)
        & np.isfinite(pth_target)
        & (pth_target >= float(np.min(pth_unique)))
        & (pth_target <= float(np.max(pth_unique)))
    )
    if np.count_nonzero(valid) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    order = np.argsort(intensity[valid])
    x_fit = intensity[valid][order]
    dt_fit = np.interp(pth_target[valid][order], pth_unique, dt_unique)
    temperature_fit = TSAI_LATTICE_TEMPERATURE_K + dt_fit

    valid_curve = np.isfinite(x_fit) & (x_fit > 0) & np.isfinite(temperature_fit) & (temperature_fit > 0)
    return x_fit[valid_curve], temperature_fit[valid_curve]


def _compute_linear_fit(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[float, float, float] | None:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return None

    x = x[valid]
    y = y[valid]
    if np.allclose(x, x[0]):
        return None

    slope, intercept = np.polyfit(x, y, deg=1)
    y_fit = slope * x + intercept
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(intercept), float(r2)


def _build_linear_fit_curve(
    x_values: np.ndarray,
    slope: float,
    intercept: float,
    n_points: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_values, dtype=float)
    valid = np.isfinite(x) & (x > 0)
    if np.count_nonzero(valid) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    x_min = float(np.min(x[valid]))
    x_max = float(np.max(x[valid]))
    if x_max <= x_min:
        return np.array([], dtype=float), np.array([], dtype=float)

    x_fit = np.linspace(x_min, x_max, int(n_points), dtype=float)
    y_fit = slope * x_fit + intercept
    valid_curve = np.isfinite(x_fit) & np.isfinite(y_fit)
    return x_fit[valid_curve], y_fit[valid_curve]


def _format_tsai_q_fit_summary(
    label: str,
    fit_result: tuple[float, float, float] | None,
) -> str | None:
    if fit_result is None:
        return None
    q_factor, _intercept, r2 = fit_result
    return f"{label}: Q={q_factor:.2e} W cm^-3 K^-1, R^2={r2:.3f}"


def plot_raw_spectra(
    energy_ev: np.ndarray,
    spectra: np.ndarray,
    intensities_w_cm2: np.ndarray,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=_page_single_column_figsize(PAGE_SHORT_FIG_HEIGHT_IN))
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
    _style_colorbar(cbar, r"Excitation intensity (W cm$^{-2}$)")
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.11, top=0.92)
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
        figsize=_page_single_column_figsize(PAGE_TALL_FIG_HEIGHT_IN),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.05},
    )
    ax0, ax1 = axes

    scan_fill_color = "#ffd166"
    scan_edge_color = "#c97b00"
    selected_fill_color = "#80ed99"
    selected_edge_color = "#1b7f3b"
    envelope_fill_color = "#74c0fc"
    envelope_edge_color = "#0b5ed7"

    ax0.plot(energy_ev, intensity, color="#1f4e79", lw=1.8, label="Experiment", zorder=5)
    ax0.plot(
        energy_ev,
        intensity_model,
        color="#c45a00",
        lw=1.4,
        ls="-",
        label="GPL model",
        zorder=6,
    )

    style_axes(ax0, logy=True)
    positive_intensity = intensity[np.isfinite(intensity) & (intensity > 0)]
    if positive_intensity.size > 0:
        y_top = 10.0 ** (float(np.ceil(np.log10(np.max(positive_intensity)))) + 1.0)
        y_bottom = 10.0 ** float(np.floor(np.log10(np.min(positive_intensity))))
        if np.isfinite(y_top) and np.isfinite(y_bottom) and (y_top > y_bottom):
            ax0.set_ylim(y_bottom, y_top)
    ax0.tick_params(axis="x", which="both", labelbottom=False)
    ax0.set_ylabel(r"PL intensity, $I_{\mathrm{PL}}$ (a.u.)")
    ax0.legend(loc="lower left", fontsize=LEGEND_FONT_SIZE, frameon=False)
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
    info_artist = ax0.text(
        0.985,
        0.97,
        info_text,
        transform=ax0.transAxes,
        ha="right",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        clip_on=False,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "boxstyle": "square,pad=0.25",
            "alpha": 0.84,
        },
    )
    _raise_annotation(info_artist)

    y_all = linearized_signal(energy_ev[intensity > 0], intensity[intensity > 0])
    if scan_domain_ev is not None:
        ax1.axvspan(
            scan_domain_ev[0],
            scan_domain_ev[1],
            color=scan_fill_color,
            alpha=0.14,
            zorder=0,
            label="Full scan domain",
        )
        ax1.axvline(
            scan_domain_ev[0], color=scan_edge_color, lw=1.2, ls=":", alpha=0.95, zorder=1
        )
        ax1.axvline(
            scan_domain_ev[1], color=scan_edge_color, lw=1.2, ls=":", alpha=0.95, zorder=1
        )
    if fit_range_windows_ev:
        for lo_ev, hi_ev in fit_range_windows_ev:
            ax1.hlines(
                y=0.06,
                xmin=lo_ev,
                xmax=hi_ev,
                transform=ax1.get_xaxis_transform(),
                color=envelope_edge_color,
                lw=1.0,
                alpha=0.28,
                zorder=1,
            )
        lo_env = float(min(w[0] for w in fit_range_windows_ev))
        hi_env = float(max(w[1] for w in fit_range_windows_ev))
        coverage_pct = 100.0 * FIT_RANGE_SCAN_PLOT_WEIGHT_COVERAGE
        ax1.axvspan(
            lo_env,
            hi_env,
            facecolor=envelope_fill_color,
            alpha=0.16,
            hatch="///",
            edgecolor=envelope_edge_color,
            lw=1.0,
            zorder=1,
            label=f"{coverage_pct:.0f}% AICc-weight window envelope",
        )
        ax1.axvline(lo_env, color=envelope_edge_color, lw=1.1, ls="--", alpha=0.9, zorder=2)
        ax1.axvline(hi_env, color=envelope_edge_color, lw=1.1, ls="--", alpha=0.9, zorder=2)
    ax1.axvspan(
        fit_min_ev,
        fit_max_ev,
        facecolor=selected_fill_color,
        alpha=0.10,
        hatch="xx",
        edgecolor=selected_edge_color,
        lw=1.25,
        zorder=3,
        label="Selected fit window",
    )
    ax1.axvline(fit_min_ev, color=selected_edge_color, lw=1.35, ls="-", alpha=0.98, zorder=4)
    ax1.axvline(fit_max_ev, color=selected_edge_color, lw=1.35, ls="-", alpha=0.98, zorder=4)
    ax1.plot(energy_ev[intensity > 0], y_all, color="0.35", lw=1.0, label="Linearized data")

    x_fit_ev = energy_ev[fit_mask]
    x_fit_j = x_fit_ev * E_CHARGE
    y_line = result.slope * x_fit_j + result.intercept
    ax1.plot(
        x_fit_ev,
        y_line,
        color="#d62828",
        lw=1.45,
        ls="--",
        label="Linear regression",
    )
    style_axes(ax1)
    ax1.set_xlabel(r"Photon energy, $E$ (eV)")
    ax1.set_ylabel(r"$\ln\!\left(\frac{h^3 c^2}{2E^2}I_{\mathrm{PL}}\right)$")
    ax1.legend(loc="lower left", fontsize=LEGEND_FONT_SIZE, frameon=False)

    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.09, top=0.94, hspace=0.04)
    save_figure(fig, outpath)
    plt.close(fig)


def plot_summary(results_df: pd.DataFrame, outpath: Path) -> None:
    x = results_df["intensity_w_cm2"].to_numpy(dtype=float)
    x_valid = x[np.isfinite(x) & (x > 0)]
    if x_valid.size > 0:
        # Keep a small visual padding while using the full log-x width of the data range.
        x_pad_factor = 10 ** 0.02
        x_min_plot = float(np.min(x_valid) / x_pad_factor)
        x_max_plot = float(np.max(x_valid) * x_pad_factor)
    else:
        x_min_plot = np.nan
        x_max_plot = np.nan

    n_vals = results_df["carrier_density_cm3"].to_numpy(dtype=float)
    n_err = results_df["carrier_density_err_total_cm3"].to_numpy(dtype=float)
    n_fd_vals = results_df["carrier_density_fd_cm3"].to_numpy(dtype=float)
    n_fd_err = results_df["carrier_density_fd_err_total_cm3"].to_numpy(dtype=float)
    temperature_k = results_df["temperature_k"].to_numpy(dtype=float)
    temperature_fit = _compute_linear_fit(x, temperature_k)
    if temperature_fit is None:
        x_temperature_fit = np.array([], dtype=float)
        t_temperature_fit = np.array([], dtype=float)
    else:
        x_temperature_fit, t_temperature_fit = _build_linear_fit_curve(
            x_values=x,
            slope=temperature_fit[0],
            intercept=temperature_fit[1],
        )

    def _set_common_intensity_xlim(ax: plt.Axes) -> None:
        if np.isfinite(x_min_plot) and np.isfinite(x_max_plot) and (x_max_plot > x_min_plot):
            ax.set_xlim(x_min_plot, x_max_plot)

    def _plot_temperature_panel(
        _fig: plt.Figure,
        ax: plt.Axes,
        *,
        show_xlabel: bool,
        show_panel_label: bool,
    ) -> None:
        ax.errorbar(
            x,
            temperature_k,
            yerr=results_df["temperature_err_total_k"],
            fmt="o",
            linestyle="none",
            ms=4.5,
            capsize=2.5,
            elinewidth=1.0,
            color="#1565c0",
        )
        if x_temperature_fit.size >= 2:
            ax.plot(
                x_temperature_fit,
                t_temperature_fit,
                color="#263238",
                lw=1.2,
                ls="--",
                alpha=0.92,
                label="Linear fit",
                zorder=2,
            )
        style_axes(ax, logx=True)
        ax.set_ylabel(r"Temperature, $T$ (K)")
        if show_xlabel:
            ax.set_xlabel(r"Excitation intensity, $I_{exc}$ (W cm$^{-2}$)")
        else:
            ax.tick_params(labelbottom=False)
        _set_common_intensity_xlim(ax)
        if temperature_fit is not None:
            fit_text_artist = ax.text(
                0.03,
                0.97,
                f"Linear fit: T = ({temperature_fit[0]:.3e}) I + ({temperature_fit[1]:.2f})\n"
                + rf"$R^2$ = {temperature_fit[2]:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=ANNOTATION_FONT_SIZE,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "boxstyle": "square,pad=0.2",
                    "alpha": 0.84,
                },
            )
            _raise_annotation(fit_text_artist)
        if x_temperature_fit.size >= 2:
            ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE, frameon=False)
        if show_panel_label:
            _add_panel_label(ax, "(a)")

    def _plot_qfls_panel(
        _fig: plt.Figure,
        ax: plt.Axes,
        *,
        show_xlabel: bool,
        show_panel_label: bool,
    ) -> None:
        ax.errorbar(
            x,
            results_df["qfls_ev"],
            yerr=results_df["qfls_err_total_ev"],
            fmt="o",
            linestyle="none",
            ms=4.5,
            capsize=2.5,
            elinewidth=1.0,
            color="#6a1b9a",
            label=r"$\Delta\mu$",
        )
        ax.errorbar(
            x,
            results_df["qfls_effective_ev"],
            yerr=results_df["qfls_effective_err_total_ev"],
            fmt="s",
            linestyle="none",
            ms=3.5,
            capsize=2.0,
            elinewidth=0.9,
            color="#9c27b0",
            alpha=0.82,
            label=r"$\Delta\mu_{\mathrm{eff}}$",
        )
        style_axes(ax, logx=True)
        ax.set_ylabel(r"QFLS (eV)")
        if show_xlabel:
            ax.set_xlabel(r"Excitation intensity, $I_{exc}$ (W cm$^{-2}$)")
        else:
            ax.tick_params(labelbottom=False)
        ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE, frameon=False)
        _set_common_intensity_xlim(ax)
        if show_panel_label:
            _add_panel_label(ax, "(b)")

    def _plot_chemical_potential_panel(
        _fig: plt.Figure,
        ax: plt.Axes,
        *,
        show_xlabel: bool,
        show_panel_label: bool,
    ) -> None:
        ax.errorbar(
            x,
            results_df["mu_e_ev"],
            yerr=results_df["mu_e_err_total_ev"],
            fmt="o",
            linestyle="none",
            ms=4.3,
            capsize=2.5,
            elinewidth=1.0,
            color="#ef6c00",
            label=r"$\mu_e$ (MB)",
        )
        ax.errorbar(
            x,
            results_df["mu_h_ev"],
            yerr=results_df["mu_h_err_total_ev"],
            fmt="o",
            linestyle="none",
            ms=4.3,
            capsize=2.5,
            elinewidth=1.0,
            color="#2e7d32",
            label=r"$\mu_h$ (MB)",
        )
        ax.errorbar(
            x,
            results_df["mu_e_fd_ev"],
            yerr=results_df["mu_e_fd_err_total_ev"],
            fmt="s",
            linestyle="none",
            ms=3.9,
            capsize=2.0,
            elinewidth=0.9,
            color="#bf360c",
            alpha=0.9,
            label=r"$\mu_e$ (FD)",
        )
        ax.errorbar(
            x,
            results_df["mu_h_fd_ev"],
            yerr=results_df["mu_h_fd_err_total_ev"],
            fmt="s",
            linestyle="none",
            ms=3.9,
            capsize=2.0,
            elinewidth=0.9,
            color="#1b5e20",
            alpha=0.9,
            label=r"$\mu_h$ (FD)",
        )
        style_axes(ax, logx=True)
        ax.set_ylabel(r"Chemical potential (eV)")
        if show_xlabel:
            ax.set_xlabel(r"Excitation intensity, $I_{exc}$ (W cm$^{-2}$)")
        else:
            ax.tick_params(labelbottom=False)
        ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE, frameon=False)
        _set_common_intensity_xlim(ax)
        if show_panel_label:
            _add_panel_label(ax, "(c)")

    def _plot_carrier_density_panel(
        _fig: plt.Figure,
        ax: plt.Axes,
        *,
        show_xlabel: bool,
        show_panel_label: bool,
    ) -> None:
        ax.errorbar(
            x,
            n_vals,
            yerr=_safe_log_yerr(y=n_vals, err=n_err),
            fmt="o",
            linestyle="none",
            ms=4.5,
            capsize=2.5,
            elinewidth=1.0,
            color="#00838f",
            label=r"$n$ (MB)",
        )
        ax.errorbar(
            x,
            n_fd_vals,
            yerr=_safe_log_yerr(y=n_fd_vals, err=n_fd_err),
            fmt="s",
            linestyle="none",
            ms=4.0,
            capsize=2.0,
            elinewidth=0.9,
            color="#004d40",
            alpha=0.9,
            label=r"$n$ (FD)",
        )
        style_axes(ax, logx=True, logy=True)
        ax.set_ylabel(r"Carrier density, $n$ (cm$^{-3}$)")
        if show_xlabel:
            ax.set_xlabel(r"Excitation intensity, $I_{exc}$ (W cm$^{-2}$)")
        else:
            ax.tick_params(labelbottom=False)
        ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE, frameon=False)
        _set_common_intensity_xlim(ax)
        if show_panel_label:
            _add_panel_label(ax, "(d)")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=_page_single_column_figsize(PAGE_GRID_FIG_HEIGHT_IN),
        sharex=True,
    )
    ax00, ax01, ax10, ax11 = axes.ravel()
    _plot_temperature_panel(fig, ax00, show_xlabel=False, show_panel_label=True)
    _plot_qfls_panel(fig, ax01, show_xlabel=False, show_panel_label=True)
    _plot_chemical_potential_panel(fig, ax10, show_xlabel=True, show_panel_label=True)
    _plot_carrier_density_panel(fig, ax11, show_xlabel=True, show_panel_label=True)

    fig.suptitle(
        "Extracted hot-carrier parameters versus excitation intensity",
        y=0.988,
        fontsize=SUPTITLE_FONT_SIZE,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.10, top=0.935, hspace=0.07, wspace=0.22)
    save_figure(fig, outpath)
    plt.close(fig)

    panel_dir = _presentation_plot_dir(outpath)
    _export_single_panel_figure(
        panel_dir,
        "temperature_vs_intensity",
        lambda fig, ax: _plot_temperature_panel(fig, ax, show_xlabel=True, show_panel_label=False),
        title="Carrier temperature",
    )
    _export_single_panel_figure(
        panel_dir,
        "qfls_vs_intensity",
        lambda fig, ax: _plot_qfls_panel(fig, ax, show_xlabel=True, show_panel_label=False),
        title="Quasi-Fermi level splitting",
    )
    _export_single_panel_figure(
        panel_dir,
        "chemical_potentials_vs_intensity",
        lambda fig, ax: _plot_chemical_potential_panel(fig, ax, show_xlabel=True, show_panel_label=False),
        title="Carrier chemical potentials",
    )
    _export_single_panel_figure(
        panel_dir,
        "carrier_density_vs_intensity",
        lambda fig, ax: _plot_carrier_density_panel(fig, ax, show_xlabel=True, show_panel_label=False),
        title="Carrier density",
    )


def plot_temperature_vs_thermalized_power(results_df: pd.DataFrame, outpath: Path) -> None:
    p_th_w_cm3 = results_df.get(
        "thermalized_power_w_cm3",
        pd.Series(np.full(results_df.shape[0], np.nan, dtype=float), index=results_df.index),
    ).to_numpy(dtype=float)
    p_th_err_w_cm3 = _sanitize_nonnegative(
        results_df.get(
            "thermalized_power_err_w_cm3",
            pd.Series(np.zeros(results_df.shape[0], dtype=float), index=results_df.index),
        ).to_numpy(dtype=float)
    )
    temperature_k = results_df["temperature_k"].to_numpy(dtype=float)
    temperature_err_k = _sanitize_nonnegative(
        results_df.get(
            "temperature_err_total_k",
            pd.Series(np.zeros(results_df.shape[0], dtype=float), index=results_df.index),
        ).to_numpy(dtype=float)
    )
    temperature_rise_k = temperature_k - TSAI_LATTICE_TEMPERATURE_K

    valid = (
        np.isfinite(p_th_w_cm3)
        & (p_th_w_cm3 > 0)
        & np.isfinite(temperature_k)
        & (temperature_k > 0)
    )
    if np.count_nonzero(valid) < 2:
        return

    p_th_plot = p_th_w_cm3[valid]
    p_th_err_plot = p_th_err_w_cm3[valid]
    temperature_plot = temperature_k[valid]
    temperature_err_plot = temperature_err_k[valid]
    temperature_rise_plot = temperature_rise_k[valid]

    linear_fit = _compute_linear_fit(p_th_plot, temperature_plot)
    if linear_fit is None:
        x_linear_fit = np.array([], dtype=float)
        y_linear_fit = np.array([], dtype=float)
    else:
        x_linear_fit, y_linear_fit = _build_linear_fit_curve(
            x_values=p_th_plot,
            slope=linear_fit[0],
            intercept=linear_fit[1],
        )

    fig, ax = plt.subplots(figsize=_page_single_column_figsize(PAGE_MEDIUM_FIG_HEIGHT_IN))
    ax.errorbar(
        p_th_plot,
        temperature_plot,
        xerr=p_th_err_plot,
        yerr=temperature_err_plot,
        fmt="o",
        linestyle="none",
        ms=4.5,
        capsize=2.5,
        elinewidth=1.0,
        color="#1565c0",
    )
    if x_linear_fit.size >= 2:
        ax.plot(
            x_linear_fit,
            y_linear_fit,
            color="#263238",
            lw=1.2,
            ls="--",
            alpha=0.92,
            label="Linear fit",
            zorder=2,
        )

    style_axes(ax, logx=True)
    ax.set_xlabel(r"Thermalized power, $P_{\mathrm{th}}$ (W cm$^{-3}$)")
    ax.set_ylabel(r"Temperature, $T$ (K)")

    text_lines: list[str] = []
    if linear_fit is not None:
        text_lines.append(
            f"Linear fit: T = ({linear_fit[0]:.3e}) P_th + ({linear_fit[1]:.2f})"
        )
        text_lines.append(rf"$R^2$ = {linear_fit[2]:.3f}")
    if text_lines:
        fit_text_artist = ax.text(
            0.03,
            0.97,
            "\n".join(text_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=ANNOTATION_FONT_SIZE,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "boxstyle": "square,pad=0.2",
                "alpha": 0.84,
            },
        )
        _raise_annotation(fit_text_artist)

    if x_linear_fit.size >= 2:
        ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE, frameon=False)

    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.92)
    save_figure(fig, outpath)
    plt.close(fig)


def plot_thermalized_power_diagnostics(results_df: pd.DataFrame, outpath: Path) -> None:
    n_mb_cm3 = results_df["carrier_density_cm3"].to_numpy(dtype=float)
    n_mb_err_cm3 = _sanitize_nonnegative(
        results_df.get(
            "carrier_density_err_total_cm3",
            pd.Series(np.zeros(results_df.shape[0], dtype=float)),
        ).to_numpy(dtype=float)
    )
    n_fd_cm3 = results_df.get(
        "carrier_density_fd_cm3",
        pd.Series(n_mb_cm3, index=results_df.index),
    ).to_numpy(dtype=float)
    n_fd_err_cm3 = _sanitize_nonnegative(
        results_df.get(
            "carrier_density_fd_err_total_cm3",
            pd.Series(n_mb_err_cm3, index=results_df.index),
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
    default_model = "mb"
    if (
        "power_balance_carrier_statistics_model" in results_df.columns
        and (not results_df.empty)
    ):
        default_model = str(
            results_df["power_balance_carrier_statistics_model"].iloc[0]
        ).strip().lower()
    p_th_per_carrier_mb_ev_s = results_df.get(
        "thermalized_power_per_carrier_mb_ev_s",
        results_df.get(
            "thermalized_power_per_carrier_ev_s",
            pd.Series(np.full(results_df.shape[0], np.nan, dtype=float)),
        ),
    ).to_numpy(dtype=float)
    p_th_per_carrier_mb_err_ev_s = _sanitize_nonnegative(
        results_df.get(
            "thermalized_power_per_carrier_mb_err_ev_s",
            results_df.get(
                "thermalized_power_per_carrier_err_ev_s",
                pd.Series(np.zeros(results_df.shape[0], dtype=float)),
            ),
        ).to_numpy(dtype=float)
    )
    p_th_per_carrier_fd_ev_s = results_df.get(
        "thermalized_power_per_carrier_fd_ev_s",
        results_df.get(
            "thermalized_power_per_carrier_ev_s",
            pd.Series(np.full(results_df.shape[0], np.nan, dtype=float)),
        ),
    ).to_numpy(dtype=float)
    p_th_per_carrier_fd_err_ev_s = _sanitize_nonnegative(
        results_df.get(
            "thermalized_power_per_carrier_fd_err_ev_s",
            results_df.get(
                "thermalized_power_per_carrier_err_ev_s",
                pd.Series(np.zeros(results_df.shape[0], dtype=float)),
            ),
        ).to_numpy(dtype=float)
    )
    thermalized_energy_pair_ev = results_df["thermalized_energy_per_pair_ev"].to_numpy(dtype=float)
    intensity_w_cm2 = results_df["intensity_w_cm2"].to_numpy(dtype=float)

    valid_common = (
        np.isfinite(temperature_k)
        & np.isfinite(p_th_w_cm3)
        & np.isfinite(thermalized_energy_pair_ev)
        & (temperature_k > 0)
        & (p_th_w_cm3 > 0)
    )
    if np.count_nonzero(valid_common) < 3:
        return

    valid_mb_state = valid_common & np.isfinite(n_mb_cm3) & (n_mb_cm3 > 0)
    valid_fd_state = valid_common & np.isfinite(n_fd_cm3) & (n_fd_cm3 > 0)
    valid_mb_per_carrier = (
        valid_mb_state
        & np.isfinite(p_th_per_carrier_mb_ev_s)
        & (p_th_per_carrier_mb_ev_s > 0)
    )
    valid_fd_per_carrier = (
        valid_fd_state
        & np.isfinite(p_th_per_carrier_fd_ev_s)
        & (p_th_per_carrier_fd_ev_s > 0)
    )
    valid_thermal = (
        valid_common
        & np.isfinite(intensity_w_cm2)
        & (intensity_w_cm2 > 0)
    )
    if (
        max(np.count_nonzero(valid_mb_state), np.count_nonzero(valid_fd_state)) < 3
        or np.count_nonzero(valid_thermal) < 3
    ):
        return

    temp_all = temperature_k[valid_common]
    temp_norm = Normalize(
        vmin=float(np.min(temp_all)),
        vmax=float(np.max(temp_all) + (1e-9 if np.max(temp_all) == np.min(temp_all) else 0.0)),
    )
    temp_cmap = "viridis"
    fd_label_context = "default" if default_model == "fd" else "comparison"
    left_column_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="0.35",
            markersize=5.5,
            label="MB-state estimate",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor="0.25",
            markeredgecolor="black",
            markersize=6.0,
            label=f"FD-state estimate ({fd_label_context})",
        ),
    ]
    density_points = np.concatenate([n_mb_cm3[valid_mb_state], n_fd_cm3[valid_fd_state]])
    density_points = density_points[np.isfinite(density_points) & (density_points > 0)]
    density_min_plot = np.nan
    density_max_plot = np.nan
    if density_points.size > 0:
        density_pad_factor = 10 ** 0.05
        density_min_plot = float(np.min(density_points) / density_pad_factor)
        density_max_plot = float(np.max(density_points) * density_pad_factor)

    n_default = n_fd_cm3 if default_model == "fd" else n_mb_cm3
    valid_default = valid_fd_state if default_model == "fd" else valid_mb_state
    n_default_plot = n_default[valid_default]
    norm_n = LogNorm(
        vmin=float(np.min(n_default_plot)),
        vmax=float(np.max(n_default_plot) + (1e-9 if np.max(n_default_plot) == np.min(n_default_plot) else 0.0)),
    )

    intensity_norm = LogNorm(
        vmin=float(np.min(intensity_w_cm2[valid_thermal])),
        vmax=float(
            np.max(intensity_w_cm2[valid_thermal])
            + (
                1e-9
                if np.max(intensity_w_cm2[valid_thermal]) == np.min(intensity_w_cm2[valid_thermal])
                else 0.0
            )
        ),
    )
    e_laser_ev = float(results_df["laser_photon_energy_ev"].to_numpy(dtype=float)[0])
    eta_values = results_df["plqy_eta"].to_numpy(dtype=float)
    eta_finite = eta_values[np.isfinite(eta_values)]
    if eta_finite.size == 0:
        eta_finite = np.array([0.0], dtype=float)
    eta_min = float(np.min(eta_finite))
    eta_max = float(np.max(eta_finite))
    t_line = np.linspace(
        float(np.min(temperature_k[valid_thermal])) * 0.98,
        float(np.max(temperature_k[valid_thermal])) * 1.02,
        160,
    )
    temperature_points = np.concatenate([temperature_k[valid_default], temperature_k[valid_thermal]])
    temperature_points = temperature_points[np.isfinite(temperature_points)]
    temperature_xlim: tuple[float, float] | None = None
    if temperature_points.size > 0:
        temp_span = float(np.max(temperature_points) - np.min(temperature_points))
        temp_pad = 3.0 if temp_span <= 0 else max(2.5, 0.06 * temp_span)
        temperature_xlim = (
            float(np.min(temperature_points) - temp_pad),
            float(np.max(temperature_points) + temp_pad),
        )

    def _set_density_xlim(ax: plt.Axes) -> None:
        if np.isfinite(density_min_plot) and np.isfinite(density_max_plot) and (density_max_plot > density_min_plot):
            ax.set_xlim(density_min_plot, density_max_plot)

    def _set_temperature_xlim(ax: plt.Axes) -> None:
        if temperature_xlim is not None and temperature_xlim[1] > temperature_xlim[0]:
            ax.set_xlim(*temperature_xlim)

    def _plot_density_power_panel(
        fig: plt.Figure,
        ax: plt.Axes,
        *,
        show_xlabel: bool,
        show_panel_label: bool,
    ) -> None:
        if np.count_nonzero(valid_mb_state) > 0:
            ax.errorbar(
                n_mb_cm3[valid_mb_state],
                p_th_w_cm3[valid_mb_state],
                xerr=n_mb_err_cm3[valid_mb_state],
                yerr=_safe_log_yerr(
                    y=p_th_w_cm3[valid_mb_state],
                    err=p_th_err_w_cm3[valid_mb_state],
                ),
                fmt="none",
                ecolor="#90a4ae",
                alpha=0.22,
                elinewidth=0.7,
                capsize=1.5,
                zorder=1,
            )
            ax.scatter(
                n_mb_cm3[valid_mb_state],
                p_th_w_cm3[valid_mb_state],
                c=temperature_k[valid_mb_state],
                cmap=temp_cmap,
                norm=temp_norm,
                s=42,
                marker="o",
                edgecolors="white",
                linewidths=0.45,
                alpha=0.58,
                zorder=2,
            )
        if np.count_nonzero(valid_fd_state) > 0:
            ax.errorbar(
                n_fd_cm3[valid_fd_state],
                p_th_w_cm3[valid_fd_state],
                xerr=n_fd_err_cm3[valid_fd_state],
                yerr=_safe_log_yerr(
                    y=p_th_w_cm3[valid_fd_state],
                    err=p_th_err_w_cm3[valid_fd_state],
                ),
                fmt="none",
                ecolor="#455a64",
                alpha=0.28,
                elinewidth=0.8,
                capsize=1.7,
                zorder=2,
            )
            ax.scatter(
                n_fd_cm3[valid_fd_state],
                p_th_w_cm3[valid_fd_state],
                c=temperature_k[valid_fd_state],
                cmap=temp_cmap,
                norm=temp_norm,
                s=58,
                marker="^",
                edgecolors="black",
                linewidths=0.4,
                alpha=0.92,
                zorder=3,
            )
        style_axes(ax, logx=True, logy=True)
        ax.set_ylabel(r"$P_{\mathrm{th}}$ (W cm$^{-3}$)")
        if show_xlabel:
            ax.set_xlabel(r"Carrier density, $n$ (cm$^{-3}$)")
        else:
            ax.tick_params(labelbottom=False)
        ax.legend(
            handles=left_column_handles,
            loc="lower right",
            fontsize=LEGEND_FONT_SIZE,
            frameon=False,
        )
        cbar = fig.colorbar(
            cm.ScalarMappable(norm=temp_norm, cmap=temp_cmap),
            ax=ax,
            pad=0.016,
            fraction=0.045,
        )
        _style_colorbar(cbar, r"$T$ (K)")
        _set_density_xlim(ax)
        if show_panel_label:
            _add_panel_label(ax, "(a)")

    def _plot_temperature_power_panel(
        fig: plt.Figure,
        ax: plt.Axes,
        *,
        show_xlabel: bool,
        show_panel_label: bool,
    ) -> None:
        ax.errorbar(
            temperature_k[valid_default],
            p_th_w_cm3[valid_default],
            xerr=temperature_err_k[valid_default],
            yerr=_safe_log_yerr(y=p_th_w_cm3[valid_default], err=p_th_err_w_cm3[valid_default]),
            fmt="none",
            ecolor="0.6",
            alpha=0.35,
            elinewidth=0.8,
            capsize=1.8,
            zorder=1,
        )
        scatter = ax.scatter(
            temperature_k[valid_default],
            p_th_w_cm3[valid_default],
            c=n_default_plot,
            cmap="cividis",
            norm=norm_n,
            s=56,
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
        )
        style_axes(ax, logy=True)
        ax.set_ylabel(r"$P_{\mathrm{th}}$ (W cm$^{-3}$)")
        if show_xlabel:
            ax.set_xlabel(r"Carrier temperature, $T$ (K)")
        else:
            ax.tick_params(labelbottom=False)
        cbar = fig.colorbar(scatter, ax=ax, pad=0.016, fraction=0.042)
        _style_colorbar(cbar, r"$n$ (cm$^{-3}$)")
        _set_temperature_xlim(ax)
        if show_panel_label:
            _add_panel_label(ax, "(b)")

    def _plot_per_carrier_panel(
        fig: plt.Figure,
        ax: plt.Axes,
        *,
        show_xlabel: bool,
        show_panel_label: bool,
    ) -> None:
        if np.count_nonzero(valid_mb_per_carrier) > 0:
            ax.errorbar(
                n_mb_cm3[valid_mb_per_carrier],
                p_th_per_carrier_mb_ev_s[valid_mb_per_carrier],
                xerr=n_mb_err_cm3[valid_mb_per_carrier],
                yerr=_safe_log_yerr(
                    y=p_th_per_carrier_mb_ev_s[valid_mb_per_carrier],
                    err=p_th_per_carrier_mb_err_ev_s[valid_mb_per_carrier],
                ),
                fmt="none",
                ecolor="#8e24aa",
                alpha=0.22,
                elinewidth=0.7,
                capsize=1.5,
                zorder=1,
            )
            ax.scatter(
                n_mb_cm3[valid_mb_per_carrier],
                p_th_per_carrier_mb_ev_s[valid_mb_per_carrier],
                c=temperature_k[valid_mb_per_carrier],
                cmap=temp_cmap,
                norm=temp_norm,
                s=42,
                marker="o",
                edgecolors="white",
                linewidths=0.45,
                alpha=0.58,
                zorder=2,
            )
        if np.count_nonzero(valid_fd_per_carrier) > 0:
            ax.errorbar(
                n_fd_cm3[valid_fd_per_carrier],
                p_th_per_carrier_fd_ev_s[valid_fd_per_carrier],
                xerr=n_fd_err_cm3[valid_fd_per_carrier],
                yerr=_safe_log_yerr(
                    y=p_th_per_carrier_fd_ev_s[valid_fd_per_carrier],
                    err=p_th_per_carrier_fd_err_ev_s[valid_fd_per_carrier],
                ),
                fmt="none",
                ecolor="#4a148c",
                alpha=0.28,
                elinewidth=0.8,
                capsize=1.7,
                zorder=2,
            )
            ax.scatter(
                n_fd_cm3[valid_fd_per_carrier],
                p_th_per_carrier_fd_ev_s[valid_fd_per_carrier],
                c=temperature_k[valid_fd_per_carrier],
                cmap=temp_cmap,
                norm=temp_norm,
                s=58,
                marker="^",
                edgecolors="black",
                linewidths=0.4,
                alpha=0.92,
                zorder=3,
            )
        style_axes(ax, logx=True, logy=True)
        ax.set_ylabel(r"$P_{\mathrm{th}}/n$ (eV s$^{-1}$ carrier$^{-1}$)")
        if show_xlabel:
            ax.set_xlabel(r"Carrier density, $n$ (cm$^{-3}$)")
        else:
            ax.tick_params(labelbottom=False)
        cbar = fig.colorbar(
            cm.ScalarMappable(norm=temp_norm, cmap=temp_cmap),
            ax=ax,
            pad=0.016,
            fraction=0.045,
        )
        _style_colorbar(cbar, r"$T$ (K)")
        _set_density_xlim(ax)
        if show_panel_label:
            _add_panel_label(ax, "(c)")

    def _plot_thermalized_energy_panel(
        fig: plt.Figure,
        ax: plt.Axes,
        *,
        show_xlabel: bool,
        show_panel_label: bool,
    ) -> None:
        ax.errorbar(
            temperature_k[valid_thermal],
            thermalized_energy_pair_ev[valid_thermal],
            xerr=temperature_err_k[valid_thermal],
            fmt="none",
            ecolor="0.6",
            alpha=0.35,
            elinewidth=0.8,
            capsize=1.8,
            zorder=1,
        )
        scatter = ax.scatter(
            temperature_k[valid_thermal],
            thermalized_energy_pair_ev[valid_thermal],
            c=intensity_w_cm2[valid_thermal],
            cmap="magma",
            norm=intensity_norm,
            s=56,
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
        )
        if abs(eta_max - eta_min) < 1e-9:
            delta_e_line = e_laser_ev - (
                EG_EV + (3.0 - 2.0 * eta_min) * (K_B / E_CHARGE) * t_line
            )
            ax.plot(
                t_line,
                delta_e_line,
                "--",
                lw=1.2,
                color="0.2",
                label=r"$E_{laser}-(E_g+(3-2\eta)k_BT)$",
            )
        else:
            delta_e_lo = e_laser_ev - (
                EG_EV + (3.0 - 2.0 * eta_min) * (K_B / E_CHARGE) * t_line
            )
            delta_e_hi = e_laser_ev - (
                EG_EV + (3.0 - 2.0 * eta_max) * (K_B / E_CHARGE) * t_line
            )
            lo = np.minimum(delta_e_lo, delta_e_hi)
            hi = np.maximum(delta_e_lo, delta_e_hi)
            ax.fill_between(
                t_line,
                lo,
                hi,
                color="0.25",
                alpha=0.15,
                label=rf"Model envelope ($\eta \in [{eta_min:.3f}, {eta_max:.3f}]$)",
            )
            ax.plot(t_line, lo, "--", lw=1.0, color="0.25", alpha=0.9)
            ax.plot(t_line, hi, "--", lw=1.0, color="0.25", alpha=0.9)
        style_axes(ax)
        ax.set_ylabel(r"Thermalized energy per pair (eV)")
        if show_xlabel:
            ax.set_xlabel(r"Carrier temperature, $T$ (K)")
        else:
            ax.tick_params(labelbottom=False)
        ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE, frameon=False)
        cbar = fig.colorbar(scatter, ax=ax, pad=0.016, fraction=0.042)
        _style_colorbar(cbar, r"$I_{\mathrm{exc}}$ (W cm$^{-2}$)")
        _set_temperature_xlim(ax)
        if show_panel_label:
            _add_panel_label(ax, "(d)")

    # Use shared x-axes within each column so only the bottom row carries x labels.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.8, 7.2),
        sharex="col",
    )
    ax00, ax01 = axes[0]
    ax10, ax11 = axes[1]
    for ax in axes.ravel():
        ax.set_box_aspect(0.92)
    _plot_density_power_panel(fig, ax00, show_xlabel=False, show_panel_label=True)
    _plot_temperature_power_panel(fig, ax01, show_xlabel=False, show_panel_label=True)
    _plot_per_carrier_panel(fig, ax10, show_xlabel=True, show_panel_label=True)
    _plot_thermalized_energy_panel(fig, ax11, show_xlabel=True, show_panel_label=True)

    fig.suptitle(
        r"Thermalized-power diagnostics in carrier-state space",
        y=0.986,
        fontsize=SUPTITLE_FONT_SIZE + 0.8,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.965,
        bottom=0.09,
        top=0.93,
        hspace=0.10,
        wspace=0.24,
    )
    save_figure(fig, outpath)
    plt.close(fig)

    panel_dir = _presentation_plot_dir(outpath)
    _export_single_panel_figure(
        panel_dir,
        "thermalized_power_vs_carrier_density",
        lambda fig, ax: _plot_density_power_panel(fig, ax, show_xlabel=True, show_panel_label=False),
        title=r"Thermalized power vs carrier density",
    )
    _export_single_panel_figure(
        panel_dir,
        "thermalized_power_vs_temperature",
        lambda fig, ax: _plot_temperature_power_panel(fig, ax, show_xlabel=True, show_panel_label=False),
        title=r"Thermalized power vs temperature",
    )
    _export_single_panel_figure(
        panel_dir,
        "per_carrier_thermalized_power_vs_density",
        lambda fig, ax: _plot_per_carrier_panel(fig, ax, show_xlabel=True, show_panel_label=False),
        title=r"Per-carrier thermalized power",
    )
    _export_single_panel_figure(
        panel_dir,
        "thermalized_energy_per_pair_vs_temperature",
        lambda fig, ax: _plot_thermalized_energy_panel(fig, ax, show_xlabel=True, show_panel_label=False),
        title=r"Thermalized energy per pair",
    )


def plot_recombination_channel_contributions(
    results_df: pd.DataFrame,
    outpath: Path,
) -> None:
    required_cols = {
        "intensity_w_cm2",
        "temperature_k",
        "plqy_eta",
        "absorbed_power_w_cm3",
        "absorbed_photon_flux_cm2_s",
        "active_layer_thickness_cm",
        "nonradiative_power_w_cm3",
        "radiative_power_w_cm3",
        "recombination_power_w_cm3",
        "thermalized_power_w_cm3",
        "radiative_fraction",
        "nonradiative_fraction",
    }
    if results_df is None or results_df.empty:
        return
    if not required_cols.issubset(results_df.columns):
        return

    df = results_df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required_cols))
    df = df[(df["intensity_w_cm2"] > 0) & (df["active_layer_thickness_cm"] > 0)]
    if df.shape[0] < 2:
        return
    df = df.sort_values("intensity_w_cm2").reset_index(drop=True)

    x = df["intensity_w_cm2"].to_numpy(dtype=float)
    temperature_k = df["temperature_k"].to_numpy(dtype=float)
    eta = np.clip(df["plqy_eta"].to_numpy(dtype=float), 0.0, 1.0)
    phi_abs_cm2_s = df["absorbed_photon_flux_cm2_s"].to_numpy(dtype=float)
    thickness_cm = df["active_layer_thickness_cm"].to_numpy(dtype=float)

    p_abs_w_cm3 = df["absorbed_power_w_cm3"].to_numpy(dtype=float)
    p_nonrad_w_cm3 = df["nonradiative_power_w_cm3"].to_numpy(dtype=float)
    p_rad_w_cm3 = df["radiative_power_w_cm3"].to_numpy(dtype=float)
    p_rec_w_cm3 = df["recombination_power_w_cm3"].to_numpy(dtype=float)
    p_th_w_cm3 = df["thermalized_power_w_cm3"].to_numpy(dtype=float)

    delta_p_th_w_cm3 = np.where(
        thickness_cm > 0,
        (2.0 * eta * phi_abs_cm2_s * K_B * temperature_k) / thickness_cm,
        np.nan,
    )
    p_rec_eta0_w_cm3 = p_rec_w_cm3 + delta_p_th_w_cm3
    p_th_eta0_w_cm3 = p_th_w_cm3 - delta_p_th_w_cm3

    eta_pct = 100.0 * eta
    rad_fraction_pct = 100.0 * df["radiative_fraction"].to_numpy(dtype=float)
    nonrad_fraction_pct = 100.0 * df["nonradiative_fraction"].to_numpy(dtype=float)
    rec_rad_share_pct = 100.0 * _safe_ratio(p_rad_w_cm3, p_rec_w_cm3)
    p_th_boost_pct = 100.0 * _safe_ratio(delta_p_th_w_cm3, p_th_eta0_w_cm3)
    p_rec_reduction_pct = 100.0 * _safe_ratio(delta_p_th_w_cm3, p_rec_eta0_w_cm3)
    p_abs_correction_pct = 100.0 * _safe_ratio(delta_p_th_w_cm3, p_abs_w_cm3)

    x_pad_factor = 10 ** 0.02
    x_min_plot = float(np.min(x) / x_pad_factor)
    x_max_plot = float(np.max(x) * x_pad_factor)

    def _set_common_xlim(ax: plt.Axes) -> None:
        if x_max_plot > x_min_plot:
            ax.set_xlim(x_min_plot, x_max_plot)

    def _plot_recombination_power_panel(
        _fig: plt.Figure,
        ax: plt.Axes,
        *,
        show_xlabel: bool,
        show_panel_label: bool,
    ) -> None:
        ax.plot(
            x,
            p_nonrad_w_cm3,
            "o-",
            color="#455a64",
            lw=1.45,
            ms=4.0,
            label=r"$P_{\mathrm{nonrad}}$",
        )
        ax.plot(
            x,
            p_rad_w_cm3,
            "s-",
            color="#f4511e",
            lw=1.35,
            ms=3.8,
            label=r"$P_{\mathrm{rad}}$",
        )
        ax.plot(
            x,
            p_rec_w_cm3,
            "-",
            color="#1e88e5",
            lw=1.35,
            label=r"$P_{\mathrm{rec}} = P_{\mathrm{nonrad}} + P_{\mathrm{rad}}$",
        )
        ax.plot(
            x,
            p_rec_eta0_w_cm3,
            "--",
            color="0.20",
            lw=1.15,
            label=r"$P_{\mathrm{rec}}$ if $\eta = 0$",
        )
        ax.fill_between(
            x,
            p_rec_w_cm3,
            p_rec_eta0_w_cm3,
            color="#90caf9",
            alpha=0.18,
        )
        style_axes(ax, logx=True, logy=True)
        ax.set_ylabel(r"Power density (W cm$^{-3}$)")
        if show_xlabel:
            ax.set_xlabel(r"Excitation intensity, $I_{\mathrm{exc}}$ (W cm$^{-2}$)")
        else:
            ax.tick_params(labelbottom=False)
        ax.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE, frameon=False)
        _set_common_xlim(ax)
        if show_panel_label:
            _add_panel_label(ax, "(a)")

    def _plot_fraction_panel(
        _fig: plt.Figure,
        ax: plt.Axes,
        *,
        show_xlabel: bool,
        show_panel_label: bool,
    ) -> None:
        ax.plot(
            x,
            nonrad_fraction_pct,
            "o-",
            color="#546e7a",
            lw=1.35,
            ms=4.0,
            label=r"$P_{\mathrm{nonrad}} / P_{\mathrm{abs}}$",
        )
        ax.plot(
            x,
            rad_fraction_pct,
            "s-",
            color="#ff7043",
            lw=1.35,
            ms=3.8,
            label=r"$P_{\mathrm{rad}} / P_{\mathrm{abs}}$",
        )
        ax.plot(
            x,
            eta_pct,
            "--",
            color="#6a1b9a",
            lw=1.2,
            label=r"$\eta = \phi_{\mathrm{rad}} / \phi_{\mathrm{abs}}$",
        )
        positive_fraction_pct = np.concatenate(
            [
                eta_pct[eta_pct > 0],
                rad_fraction_pct[rad_fraction_pct > 0],
                nonrad_fraction_pct[nonrad_fraction_pct > 0],
            ]
        )
        if positive_fraction_pct.size > 0:
            y_min = max(float(np.min(positive_fraction_pct)) / 1.6, 1e-4)
            y_max = min(float(np.max(positive_fraction_pct)) * 1.35, 300.0)
            style_axes(ax, logx=True, logy=True)
            ax.set_ylim(y_min, y_max)
        else:
            style_axes(ax, logx=True)
            ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Fraction of absorbed power (%)")
        if show_xlabel:
            ax.set_xlabel(r"Excitation intensity, $I_{\mathrm{exc}}$ (W cm$^{-2}$)")
        else:
            ax.tick_params(labelbottom=False)
        ax.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE, frameon=False)
        _set_common_xlim(ax)
        if show_panel_label:
            _add_panel_label(ax, "(b)")

    def _plot_thermalized_power_panel(
        _fig: plt.Figure,
        ax: plt.Axes,
        *,
        show_xlabel: bool,
        show_panel_label: bool,
    ) -> None:
        ax.plot(
            x,
            p_th_w_cm3,
            "o-",
            color="#2e7d32",
            lw=1.45,
            ms=4.0,
            label=r"$P_{\mathrm{th}}$ with measured $\eta$",
        )
        ax.plot(
            x,
            p_th_eta0_w_cm3,
            "--",
            color="#8bc34a",
            lw=1.15,
            label=r"$P_{\mathrm{th}}$ if $\eta = 0$",
        )
        ax.fill_between(
            x,
            p_th_eta0_w_cm3,
            p_th_w_cm3,
            color="#66bb6a",
            alpha=0.18,
        )
        style_axes(ax, logx=True, logy=True)
        ax.set_ylabel(r"Power density (W cm$^{-3}$)")
        if show_xlabel:
            ax.set_xlabel(r"Excitation intensity, $I_{\mathrm{exc}}$ (W cm$^{-2}$)")
        else:
            ax.tick_params(labelbottom=False)
        ax.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE, frameon=False)
        _set_common_xlim(ax)
        if show_panel_label:
            _add_panel_label(ax, "(c)")

    def _plot_relative_change_panel(
        _fig: plt.Figure,
        ax: plt.Axes,
        *,
        show_xlabel: bool,
        show_panel_label: bool,
    ) -> None:
        ax.plot(
            x,
            p_th_boost_pct,
            "o-",
            color="#2e7d32",
            lw=1.4,
            ms=4.0,
            label=r"$\Delta P_{\mathrm{th}} / P_{\mathrm{th}}(\eta = 0)$",
        )
        ax.plot(
            x,
            p_rec_reduction_pct,
            "s--",
            color="#1e88e5",
            lw=1.2,
            ms=3.8,
            label=r"$[P_{\mathrm{rec}}(\eta = 0) - P_{\mathrm{rec}}] / P_{\mathrm{rec}}(\eta = 0)$",
        )
        ax.plot(
            x,
            p_abs_correction_pct,
            ":",
            color="#f4511e",
            lw=1.2,
            label=r"$\Delta P_{\mathrm{th}} / P_{\mathrm{abs}}$",
        )
        style_axes(ax, logx=True)
        finite_pct = np.concatenate(
            [
                p_th_boost_pct[np.isfinite(p_th_boost_pct)],
                p_rec_reduction_pct[np.isfinite(p_rec_reduction_pct)],
                p_abs_correction_pct[np.isfinite(p_abs_correction_pct)],
            ]
        )
        finite_pct = finite_pct[finite_pct >= 0]
        ax.set_ylim(
            0.0,
            float(np.max(finite_pct) * 1.18) if finite_pct.size > 0 else 1.0,
        )
        ax.set_ylabel("Relative change (%)")
        if show_xlabel:
            ax.set_xlabel(r"Excitation intensity, $I_{\mathrm{exc}}$ (W cm$^{-2}$)")
        else:
            ax.tick_params(labelbottom=False)
        ax.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE, frameon=False)
        _set_common_xlim(ax)
        if show_panel_label:
            _add_panel_label(ax, "(d)")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=_page_single_column_figsize(PAGE_LARGE_GRID_FIG_HEIGHT_IN),
        sharex=True,
    )
    ax00, ax01, ax10, ax11 = axes.ravel()
    _plot_recombination_power_panel(fig, ax00, show_xlabel=False, show_panel_label=True)
    _plot_fraction_panel(fig, ax01, show_xlabel=False, show_panel_label=True)
    _plot_thermalized_power_panel(fig, ax10, show_xlabel=True, show_panel_label=True)
    _plot_relative_change_panel(fig, ax11, show_xlabel=True, show_panel_label=True)

    fig.suptitle(
        r"Radiative versus nonradiative recombination contributions",
        y=0.988,
        fontsize=SUPTITLE_FONT_SIZE,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.935, hspace=0.08, wspace=0.20)
    save_figure(fig, outpath)
    plt.close(fig)

    panel_dir = _presentation_plot_dir(outpath)
    _export_single_panel_figure(
        panel_dir,
        "recombination_power_channels",
        lambda fig, ax: _plot_recombination_power_panel(fig, ax, show_xlabel=True, show_panel_label=False),
        title="Recombination power channels",
    )
    _export_single_panel_figure(
        panel_dir,
        "absorbed_power_fractions_and_plqy",
        lambda fig, ax: _plot_fraction_panel(fig, ax, show_xlabel=True, show_panel_label=False),
        title="Absorbed-power fractions and PLQY",
    )
    _export_single_panel_figure(
        panel_dir,
        "thermalized_power_radiative_correction",
        lambda fig, ax: _plot_thermalized_power_panel(fig, ax, show_xlabel=True, show_panel_label=False),
        title="Thermalized power with radiative correction",
    )
    _export_single_panel_figure(
        panel_dir,
        "relative_radiative_correction_size",
        lambda fig, ax: _plot_relative_change_panel(fig, ax, show_xlabel=True, show_panel_label=False),
        title="Relative size of the radiative correction",
    )


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
        figsize=_page_single_column_figsize(PAGE_MEDIUM_FIG_HEIGHT_IN),
        sharex=True,
        gridspec_kw={"height_ratios": [2.8, 1.6], "hspace": 0.06},
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
                alpha=0.24,
                zorder=0,
            )
            ax.axvline(conservative_x, color="#b71c1c", lw=1.0, ls=":", alpha=0.78)
        non_mb_text_artist = ax0.text(
            0.03,
            0.05,
            rf"Non-MB side starts near $x^* \approx {conservative_x:.2f}$",
            transform=ax0.transAxes,
            ha="left",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "boxstyle": "square,pad=0.2",
                "alpha": 0.84,
            },
        )
        _raise_annotation(non_mb_text_artist)

    style_axes(ax0)
    ax0.set_ylabel(r"$\ln\!\left(\int_{E_g}^{\infty} I_{\mathrm{PC}}(E)\,\mathrm{d}E\right)$")
    top_handles, top_labels = ax0.get_legend_handles_labels()
    top_handles.append(Line2D([0], [0], color="0.25", lw=1.2, ls="--"))
    top_labels.append("MB affine reference")
    ax0.legend(top_handles, top_labels, loc="upper left", fontsize=LEGEND_FONT_SIZE, frameon=False)

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
    ax1.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, frameon=False)
    ax1.set_xlim(x_left, x_right)

    if limits_df is not None and (not limits_df.empty):
        lines: list[str] = []
        for row in limits_df.itertuples(index=False):
            if hasattr(row, "x_limit") and np.isfinite(row.x_limit):
                lines.append(f"T={row.temperature_k:.0f} K: x*={row.x_limit:.2f}")
        if lines:
            limit_text_artist = ax1.text(
                0.03,
                0.04,
                "\n".join(lines),
                transform=ax1.transAxes,
                ha="left",
                va="bottom",
                fontsize=ANNOTATION_FONT_SIZE,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "boxstyle": "square,pad=0.2",
                    "alpha": 0.84,
                },
            )
            _raise_annotation(limit_text_artist)

    fig.suptitle(
        "Maxwell-Boltzmann validity limit from integrated generalized Planck law",
        y=0.986,
        fontsize=SUPTITLE_FONT_SIZE,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.11, top=0.94, hspace=0.08)
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

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=_page_single_column_figsize(PAGE_SHORT_FIG_HEIGHT_IN),
    )

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
    ax0.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    cbar0 = fig.colorbar(contour, ax=ax0, pad=0.02, fraction=0.05)
    _style_colorbar(cbar0, "Simulated temperature (K)")

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
    ax1.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    mae_k = float(np.nanmean(np.abs(t_sim - t_exp)))
    bias_k = float(np.nanmean(t_sim - t_exp))
    ax1.text(
        0.03,
        0.97,
        f"MAE = {mae_k:.2f} K\nBias = {bias_k:.2f} K",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        bbox={
            "facecolor": "white",
            "edgecolor": "0.35",
            "boxstyle": "square,pad=0.2",
            "alpha": 0.92,
        },
    )

    fig.suptitle(
        "Tsai-model temperature inversion against experiment",
        y=0.992,
        fontsize=SUPTITLE_FONT_SIZE,
    )
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=0.90, wspace=0.28)
    save_figure(fig, outpath)
    plt.close(fig)


def plot_tsai_temperature_rise_vs_pth_density(
    tsai_result: "TsaiWorkflowResult",
    outpath: Path,
) -> None:
    df = tsai_result.experimental_prediction_df.copy()
    default_model = str(tsai_result.delta_mu_carrier_statistics_model).strip().lower()
    dual_model_comparison = (
        tsai_result.primary_axis_name == "delta_mu_ev"
        and {
            "temperature_rise_sim_mb_k",
            "temperature_rise_sim_fd_k",
            "carrier_density_sim_mb_cm3",
            "carrier_density_sim_fd_cm3",
        }.issubset(df.columns)
    )

    if dual_model_comparison:
        needed = [
            "p_th_exp_w_cm3",
            "temperature_rise_exp_k",
            "temperature_rise_sim_mb_k",
            "temperature_rise_sim_fd_k",
            "carrier_density_exp_cm3",
            "carrier_density_sim_mb_cm3",
            "carrier_density_sim_fd_cm3",
        ]
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
        if df.shape[0] < 2:
            return

        pth = df["p_th_exp_w_cm3"].to_numpy(dtype=float)
        dT_exp = df["temperature_rise_exp_k"].to_numpy(dtype=float)
        dT_mb = df["temperature_rise_sim_mb_k"].to_numpy(dtype=float)
        dT_fd = df["temperature_rise_sim_fd_k"].to_numpy(dtype=float)
        n_exp = df["carrier_density_exp_cm3"].to_numpy(dtype=float)
        n_mb = df["carrier_density_sim_mb_cm3"].to_numpy(dtype=float)
        n_fd = df["carrier_density_sim_fd_cm3"].to_numpy(dtype=float)

        valid = (
            np.isfinite(pth)
            & (pth > 0)
            & np.isfinite(dT_exp)
            & np.isfinite(dT_mb)
            & np.isfinite(dT_fd)
            & np.isfinite(n_exp)
            & np.isfinite(n_mb)
            & np.isfinite(n_fd)
            & (n_exp > 0)
            & (n_mb > 0)
            & (n_fd > 0)
        )
        if np.count_nonzero(valid) < 2:
            return

        pth = pth[valid]
        dT_exp = dT_exp[valid]
        dT_mb = dT_mb[valid]
        dT_fd = dT_fd[valid]
        n_exp = n_exp[valid]
        n_mb = n_mb[valid]
        n_fd = n_fd[valid]
        q_fit_exp = _compute_tsai_q_factor_fit(pth, dT_exp)
        q_fit_fd = _compute_tsai_q_factor_fit(pth, dT_fd)
        q_fit_mb = _compute_tsai_q_factor_fit(pth, dT_mb)

        n_all = np.concatenate([n_exp, n_mb, n_fd])
        n_norm = LogNorm(vmin=float(np.min(n_all)), vmax=float(np.max(n_all)))
        cmap = cm.viridis

        fig, (ax0, ax1) = plt.subplots(
            2,
            1,
            figsize=_page_single_column_figsize(PAGE_TALL_FIG_HEIGHT_IN),
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.5], "hspace": 0.12},
        )
        for x, y_exp, y_fd, y_mb in zip(pth, dT_exp, dT_fd, dT_mb, strict=True):
            ax0.plot([x, x], [y_exp, y_fd], color="#37474f", lw=0.75, alpha=0.18, zorder=1)
            ax0.plot([x, x], [y_exp, y_mb], color="#8e24aa", lw=0.7, alpha=0.12, zorder=1)

        ax0.scatter(
            pth,
            dT_exp,
            c=n_exp,
            cmap=cmap,
            norm=n_norm,
            s=54,
            marker="o",
            edgecolors="white",
            linewidths=0.55,
            label="Experimental",
            zorder=3,
        )
        ax0.scatter(
            pth,
            dT_fd,
            c=n_fd,
            cmap=cmap,
            norm=n_norm,
            s=60,
            marker="^",
            edgecolors="black",
            linewidths=0.45,
            label=f"Tsai simulated (FD{' default' if default_model == 'fd' else ''})",
            zorder=4,
        )
        ax0.scatter(
            pth,
            dT_mb,
            c=n_mb,
            cmap=cmap,
            norm=n_norm,
            s=48,
            marker="s",
            edgecolors="white",
            linewidths=0.45,
            alpha=0.64,
            label=f"Tsai simulated (MB{' default' if default_model == 'mb' else ''})",
            zorder=3,
        )
        for fit_result, fit_source, color, linestyle, label in (
            (q_fit_exp, dT_exp, "#263238", "--", "Experimental Q-fit (back-transformed)"),
            (q_fit_fd, dT_fd, "#1565c0", "-.", "FD Q-fit (back-transformed)"),
            (q_fit_mb, dT_mb, "#8e24aa", ":", "MB Q-fit (back-transformed)"),
        ):
            if fit_result is None:
                continue
            fit_pth, fit_dt = _build_tsai_q_fit_curve(
                temperature_rise_k=fit_source,
                slope=fit_result[0],
                intercept=fit_result[1],
            )
            if fit_pth.size < 2:
                continue
            ax0.plot(
                fit_pth,
                fit_dt,
                color=color,
                lw=1.2,
                ls=linestyle,
                alpha=0.92,
                label=label,
                zorder=2,
            )

        style_axes(ax0, logx=True)
        ax0.set_ylabel(r"Carrier temperature rise, $T - T_L$ (K)")
        q_summary_lines = [
            line
            for line in (
                _format_tsai_q_fit_summary("Exp", q_fit_exp),
                _format_tsai_q_fit_summary("FD", q_fit_fd),
                _format_tsai_q_fit_summary("MB", q_fit_mb),
            )
            if line is not None
        ]
        if q_summary_lines:
            q_text = "Q from linearized fit:\n" + "\n".join(q_summary_lines)
            q_text_artist = ax0.text(
                0.03,
                0.97,
                q_text,
                transform=ax0.transAxes,
                ha="left",
                va="top",
                fontsize=ANNOTATION_FONT_SIZE,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "boxstyle": "square,pad=0.2",
                    "alpha": 0.84,
                },
            )
            _raise_annotation(q_text_artist)
        ax0.legend(loc="best", fontsize=LEGEND_FONT_SIZE, ncol=2, frameon=False)

        delta_t_fd = dT_fd - dT_exp
        delta_t_mb = dT_mb - dT_exp
        ax1.axhline(0.0, color="0.25", lw=1.0, ls="--", zorder=1)
        ax1.scatter(
            pth,
            delta_t_fd,
            c=n_fd,
            cmap=cmap,
            norm=n_norm,
            s=48,
            marker="^",
            edgecolors="black",
            linewidths=0.45,
            zorder=4,
            label="FD residual",
        )
        ax1.scatter(
            pth,
            delta_t_mb,
            c=n_mb,
            cmap=cmap,
            norm=n_norm,
            s=40,
            marker="s",
            edgecolors="white",
            linewidths=0.45,
            alpha=0.66,
            zorder=3,
            label="MB residual",
        )
        style_axes(ax1, logx=True)
        ax1.set_xlabel(r"$P_{\mathrm{th}}$ (W cm$^{-3}$)")
        ax1.set_ylabel(r"$\Delta T_{\mathrm{sim-exp}}$ (K)")
        ax1.legend(loc="best", fontsize=LEGEND_FONT_SIZE, frameon=False)

        sm = cm.ScalarMappable(norm=n_norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=[ax0, ax1], pad=0.02, fraction=0.04)
        _style_colorbar(cbar, r"Carrier density, $n$ (cm$^{-3}$)")

        mae_fd = float(np.nanmean(np.abs(delta_t_fd)))
        bias_fd = float(np.nanmean(delta_t_fd))
        mae_mb = float(np.nanmean(np.abs(delta_t_mb)))
        bias_mb = float(np.nanmean(delta_t_mb))
        residual_text_artist = ax1.text(
            0.03,
            0.96,
            f"FD: MAE={mae_fd:.2f} K, Bias={bias_fd:.2f} K\n"
            f"MB: MAE={mae_mb:.2f} K, Bias={bias_mb:.2f} K",
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=ANNOTATION_FONT_SIZE,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "boxstyle": "square,pad=0.2",
                "alpha": 0.84,
            },
        )
        _raise_annotation(residual_text_artist)

        fig.suptitle(
            r"$T-T_L$ vs $P_{\mathrm{th}}$ with MB/FD Tsai closures",
            y=0.986,
            fontsize=SUPTITLE_FONT_SIZE,
            fontweight="bold",
        )
        fig.subplots_adjust(left=0.11, right=0.89, bottom=0.10, top=0.94, hspace=0.12)
        save_figure(fig, outpath)
        plt.close(fig)
        return

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
    q_fit_exp = _compute_tsai_q_factor_fit(pth, dT_exp)
    q_fit_sim = _compute_tsai_q_factor_fit(pth, dT_sim)

    n_all = np.concatenate([n_exp, n_sim])
    n_norm = LogNorm(vmin=float(np.min(n_all)), vmax=float(np.max(n_all)))
    cmap = cm.viridis

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=_page_single_column_figsize(PAGE_TALL_FIG_HEIGHT_IN),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.5], "hspace": 0.12},
    )
    for x, y0, y1 in zip(pth, dT_exp, dT_sim, strict=True):
        ax0.plot([x, x], [y0, y1], color="0.65", lw=0.75, alpha=0.24, zorder=1)

    ax0.scatter(
        pth,
        dT_exp,
        c=n_exp,
        cmap=cmap,
        norm=n_norm,
        s=54,
        marker="o",
        edgecolors="white",
        linewidths=0.55,
        label="Experimental",
        zorder=3,
    )
    ax0.scatter(
        pth,
        dT_sim,
        c=n_sim,
        cmap=cmap,
        norm=n_norm,
        s=58,
        marker="^",
        edgecolors="black",
        linewidths=0.45,
        label="Tsai-simulated",
        zorder=4,
    )
    for fit_result, fit_source, color, linestyle, label in (
        (q_fit_exp, dT_exp, "#263238", "--", "Experimental Q-fit (back-transformed)"),
        (q_fit_sim, dT_sim, "#1565c0", "-.", "Tsai Q-fit (back-transformed)"),
    ):
        if fit_result is None:
            continue
        fit_pth, fit_dt = _build_tsai_q_fit_curve(
            temperature_rise_k=fit_source,
            slope=fit_result[0],
            intercept=fit_result[1],
        )
        if fit_pth.size < 2:
            continue
        ax0.plot(
            fit_pth,
            fit_dt,
            color=color,
            lw=1.2,
            ls=linestyle,
            alpha=0.92,
            label=label,
            zorder=2,
        )

    style_axes(ax0, logx=True)
    ax0.set_ylabel(r"Carrier temperature rise, $T - T_L$ (K)")
    ax0.legend(loc="best", fontsize=LEGEND_FONT_SIZE, frameon=False)
    q_summary_lines = [
        line
        for line in (
            _format_tsai_q_fit_summary("Exp", q_fit_exp),
            _format_tsai_q_fit_summary("Tsai", q_fit_sim),
        )
        if line is not None
    ]
    if q_summary_lines:
        q_text = "Q from linearized fit:\n" + "\n".join(q_summary_lines)
        q_text_artist = ax0.text(
            0.03,
            0.97,
            q_text,
            transform=ax0.transAxes,
            ha="left",
            va="top",
            fontsize=ANNOTATION_FONT_SIZE,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "boxstyle": "square,pad=0.2",
                "alpha": 0.84,
            },
        )
        _raise_annotation(q_text_artist)

    delta_t = dT_sim - dT_exp
    ax1.axhline(0.0, color="0.25", lw=1.0, ls="--", zorder=1)
    ax1.scatter(
        pth,
        delta_t,
        c=n_exp,
        cmap=cmap,
        norm=n_norm,
        s=44,
        marker="o",
        edgecolors="white",
        linewidths=0.45,
        zorder=3,
    )
    style_axes(ax1, logx=True)
    ax1.set_xlabel(r"$P_{\mathrm{th}}$ (W cm$^{-3}$)")
    ax1.set_ylabel(r"$\Delta T_{\mathrm{sim-exp}}$ (K)")

    sm = cm.ScalarMappable(norm=n_norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=[ax0, ax1], pad=0.02, fraction=0.04)
    _style_colorbar(cbar, r"Carrier density, $n$ (cm$^{-3}$)")

    mae_rise = float(np.nanmean(np.abs(delta_t)))
    bias_rise = float(np.nanmean(delta_t))
    residual_text_artist = ax1.text(
        0.03,
        0.96,
        f"MAE(ΔT) = {mae_rise:.2f} K\nBias(ΔT) = {bias_rise:.2f} K",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "boxstyle": "square,pad=0.2",
            "alpha": 0.84,
        },
    )
    _raise_annotation(residual_text_artist)

    fig.suptitle(
        r"$T-T_L$ vs $P_{\mathrm{th}}$ colored by carrier density",
        y=0.986,
        fontsize=SUPTITLE_FONT_SIZE,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.11, right=0.89, bottom=0.10, top=0.94, hspace=0.12)
    save_figure(fig, outpath)
    plt.close(fig)
