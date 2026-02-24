# Hot Carrier Properties Extraction from GaAs PL Spectra

This project analyzes calibrated GaAs photoluminescence (PL) spectra measured versus excitation intensity.

The code:
- extracts carrier temperature `T` and quasi-Fermi level splitting `Delta_mu` from the high-energy PL tail,
- propagates uncertainties from fit statistics, fit-range choice, and absorptivity `A0`,
- computes `mu_e`, `mu_h`, and carrier density `n` using both Maxwell-Boltzmann (MB) and Fermi-Dirac (FD) carrier statistics,
- computes thermalized/recombination power channels from excitation power using detailed balance,
- generates comparison-ready `P_th(n,T)` plots and optional experiment-vs-Tsai overlays.

Main script: `main.py`

## 1) Scientific Goal

Given multiple PL spectra at different excitation powers, estimate hot-carrier thermodynamic quantities and connect optical excitation to energy dissipation channels:
- absorbed power,
- recombination power (radiative + non-radiative),
- thermalized (cooling) power.

## 2) Physics Model

## 2.1 High-energy GPL linearization

Generalized Planck law (steady state):

`I_PL(E) ~ [2 E^2 / (h^3 c^2)] * A(E) * {exp[(E - Delta_mu)/(k_B T)] - 1}^-1`

In the high-energy tail (`E > Delta_mu + k_B T`) and with approximately constant absorptivity in the fit window (`A(E) ~ A0`):

`I_PL(E) ~ [2 E^2 / (h^3 c^2)] * A0 * exp[-(E - Delta_mu)/(k_B T)]`

Define:

`y(E) = ln[(h^3 c^2 / (2 E^2)) I_PL(E)] = m E + b`

Then:
- `m = -1/(k_B T)` -> `T = -1/(k_B m)`
- intercept carries `Delta_mu` and `ln(A0)` offset.

The code reports:
- `qfls_effective_ev`: value obtained directly from intercept (contains unknown `A0` offset),
- `qfls_ev`: corrected value using configured `ASSUMED_A0`.

## 2.2 Carrier statistics (`mu_e`, `mu_h`, `n`)

`T` and `Delta_mu` come from the GPL tail fit (Section 2.1).  
Carrier statistics are then applied as a second step, using the same `T` and `Delta_mu`.

Shared effective density-of-states terms:

`N_c(T) = 2 * [(m_e* k_B T)/(2 pi hbar^2)]^(3/2)`

`N_v(T) = 2 * [(m_h* k_B T)/(2 pi hbar^2)]^(3/2)`

### MB (analytic)

With charge neutrality (`n=p`) and `Delta_mu = mu_e + mu_h`:

`mu_e - mu_h = k_B T ln(N_c/N_v)`

`mu_e = 0.5 * [Delta_mu - k_B T ln(N_c/N_v)]`

`mu_h = 0.5 * [Delta_mu + k_B T ln(N_c/N_v)]`

`n = N_c * exp[(mu_e - E_g/2)/(k_B T)]`

### FD (numerical)

For a 3D parabolic band:

`n = N_c * F_{1/2}(eta_e)`

`p = N_v * F_{1/2}(eta_h)`

`eta_e = (mu_e - E_g/2)/(k_B T)`

`eta_h = (mu_h - E_g/2)/(k_B T)`

`Delta_mu = mu_e + mu_h`

The code solves neutrality:

`N_c * F_{1/2}(eta_e) = N_v * F_{1/2}(eta_h)`

with

`eta_h = (Delta_mu - E_g)/(k_B T) - eta_e`

by bisection, then reconstructs:

`mu_e = E_g/2 + k_B T * eta_e`

`mu_h = E_g/2 + k_B T * eta_h`

`n = N_c * F_{1/2}(eta_e)`

where `F_{1/2}` is the complete Fermi-Dirac integral.

Important interpretation:
- MB and FD differ only in the carrier-statistics back-calculation (`mu_e`, `mu_h`, `n`).
- `T` and `Delta_mu` are unchanged, because they come from the same GPL tail fit.

## 2.3 Detailed balance from excitation to thermalized power

Carrier number balance:

`phi_abs = phi_gen = phi_rad + phi_nonrad`

Energy balance:

`P_abs = P_thermalized + P_nonrad + P_rad`

With your channel energies per recombination event:

`P_nonrad = phi_nonrad * (E_g + 3 k_B T)`

`P_rad = phi_rad * (E_g + k_B T)`

So:

`P_rec = P_nonrad + P_rad`

`P_thermalized = P_abs - P_rec`

Absorbed power from excitation:

`P_abs = A(E_laser) * P_exc`

`phi_abs = P_abs / E_laser` with `E_laser = h c / lambda_laser`

Using PLQY `eta = phi_rad/(phi_rad + phi_nonrad)`:

`phi_rad = eta * phi_abs`

`phi_nonrad = (1 - eta) * phi_abs`

This is exactly what `compute_power_balance_table(...)` implements.

The code also converts area-based power (`W cm^-2`) to volumetric power (`W cm^-3`) using active-layer thickness `d`:

`P_vol = P_area / d_cm`

with `d_cm = d_nm * 1e-7`.  
For your sample, `d = 950 nm` is used by default (`ACTIVE_LAYER_THICKNESS_NM`).

## 2.4 If radiative recombination is not negligible

It is handled explicitly through `eta`:
- `eta = 0` -> purely non-radiative channel,
- larger `eta` increases radiative share and reduces average recombination energy per event from
  `E_g + 3k_B T` toward `E_g + k_B T`.

Result: for fixed `P_abs` and `T`, higher `eta` increases `P_thermalized` in this model.

## 2.5 Quantity Compared to Tsai2018

The comparison target is:

`P_th(n,T)` (thermalized/cooling power density as a function of carrier density and carrier temperature).

From CWPL, each excitation condition gives one experimental triplet:
- `n` from GPL + carrier-statistics back-calculation (`carrier_density_cm3` for MB in the current `P_th(n,T)` plot),
- `T` from GPL slope,
- `P_th` from the detailed-balance power model above (reported in `W cm^-3`).

So the experiment provides sampled points on the manifold `P_th(n,T)`; Tsai-model simulation should produce a continuous or gridded version of the same function.

## 3) What the Code Does

Pipeline in `main.py`:

1. Load `GaAs bulk_PL_avg_circle_4pixs.txt`.
2. Sort by increasing energy.
3. Plot all raw spectra (`outputs/all_spectra_logscale.png`).
4. Per spectrum:
   - build candidate high-energy scan domain,
   - enumerate contiguous candidate windows,
   - fit each candidate in linearized space,
   - keep physically plausible candidates,
   - select primary window by minimum AICc,
   - fit selected window and extract `T`, `Delta_mu`, then compute (`mu_e`, `mu_h`, `n`) with MB and FD,
   - generate diagnostic plot with:
     - selected fit window,
     - full scan domain,
     - `95% AICc-weight window envelope`.
5. Aggregate all spectra into `fit_results.csv`.
6. Compute power-balance quantities (`P_abs`, `P_rec`, `P_thermalized`, fluxes, fractions) in both
   area units (`W cm^-2`) and volumetric units (`W cm^-3`) using `d=950 nm` (configurable).
7. Plot:
   - parameter trends (`outputs/parameters_vs_intensity.png`) with MB vs FD overlays for `mu_e`, `mu_h`, and `n`
   - thermalized-power diagnostics (`outputs/thermalized_power_diagnostics.png`) with
     `P_th` vs `n`, `P_th` vs `T`, `P_th/n` vs `n`, and thermalized energy per pair vs `T`
   - `P_th(n,T)` comparison figure (`outputs/pth_nT_comparison.png`)
   - if a Tsai table is provided, overlay model contours and produce parity-style comparison.

## 4) Uncertainty Model

For each extracted parameter, total uncertainty is combined in quadrature:

`sigma_total^2 = sigma_chi2^2 + sigma_range^2 + sigma_A0^2`

Components:

1. `chi2`/fit-statistical term:
- from linear regression covariance of slope/intercept,
- propagated analytically to `T`, `Delta_mu`, then via Jacobian to MB-derived `mu_e`, `mu_h`, `n`.

2. Fit-range term:
- all plausible windows in scan domain are fitted,
- each gets AICc weight (`exp(-DeltaAICc/2)`),
- parameter spread is weighted RMS around selected-window estimate.

3. `A0` term:
- high-energy absorptivity interval (`A0_HIGH_ENERGY_MIN`, `A0_HIGH_ENERGY_MAX`) defines `ASSUMED_A0` and `A0_SIGMA`,
- propagated to `Delta_mu` with
  `d(Delta_mu)/dA0 = -(k_B T)/(e A0)`,
- then to MB-derived `mu_e`, `mu_h`, `n`.

Current implementation note:
- FD quantities are reported as nominal values from the same fitted `T` and `Delta_mu`.
- Their uncertainty propagation is not yet implemented separately.

Power-balance uncertainty includes derivatives with respect to:
- `A(E_laser)`,
- `eta` (PLQY),
- fitted `T`.

## 5) Inputs That Still Need Real Experimental Values

These are currently placeholders/defaults and should be set from your measurements/calibration:

1. `ABSORPTIVITY_AT_LASER` (current value `0.6`)
- Meaning: `A(E_laser)` used in `P_abs = A(E_laser) * P_exc`.
- Must be replaced by calibrated absorptivity at laser wavelength.

2. `ABSORPTIVITY_AT_LASER_SIGMA` (default `0.0`)
- Uncertainty on `A(E_laser)`.

3. `PLQY_ETA` (default `0.0`)
- Measured PLQY in your operating conditions.
- This controls radiative vs non-radiative partition.

4. `PLQY_ETA_SIGMA` (default `0.0`)
- Uncertainty on PLQY.

Likely known from setup, but verify:

5. `LASER_WAVELENGTH_NM` (currently `532.0`)
- Should match experiment exactly.

6. `ACTIVE_LAYER_THICKNESS_NM` (currently `950.0`)
- Used to convert `W cm^-2` to `W cm^-3` for `P_th(n,T)` comparison.
- Keep consistent with the actual active region used in the Tsai-model simulation.

7. `TSAI_MODEL_TABLE_CSV` (default empty string)
- Optional CSV input for direct overlay/parity comparison.
- Required columns: `n_cm3`, `temperature_k`, `p_th_w_cm3`.

`A0` for high-energy GPL correction is already parameterized from your OptiPV interval:
- `A0_HIGH_ENERGY_MIN = 0.459`
- `A0_HIGH_ENERGY_MAX = 0.555`

## 6) How to Compare CWPL with Tsai2018

Minimal workflow:

1. Run CWPL extraction (`main.py`) to obtain experimental points `(n_i, T_i, P_th,i)`.
2. Use volumetric cooling power from CSV: `thermalized_power_w_cm3`.
3. Simulate Tsai microscopic model on a grid covering your experimental domain in `n` and `T`.
4. Export model results to CSV with columns:
   - `n_cm3`
   - `temperature_k`
   - `p_th_w_cm3`
5. Set `TSAI_MODEL_TABLE_CSV` to that file and rerun.
6. Inspect:
   - `outputs/pth_nT_comparison.png`:
     - left panel: experimental manifold `P_th(n,T)` (color map) + Tsai contours,
     - right panel: pointwise parity-style comparison at measured `(n,T)`.
   - `outputs/pth_experiment_vs_tsai.csv`:
     - includes nearest-model prediction at each experimental point and ratio `pth_ratio_tsai_over_exp`.

Interpretation strategy:
- If Tsai contours align with the experimental point cloud in `(n,T)` space, the model captures the cooling manifold shape.
- If parity points cluster around the 1:1 line, model magnitude is consistent.
- Systematic ratio trends vs intensity, `n`, or `T` indicate missing physics/parameter mismatch (e.g. PLQY, absorption, hot-phonon lifetime assumptions).

## 7) Outputs

Generated in `outputs/`:

- `fit_results.csv`:
  - fitted PL parameters (`T`, `Delta_mu`)
  - MB carrier outputs: `mu_e_ev`, `mu_h_ev`, `carrier_density_cm3`
  - FD carrier outputs: `mu_e_fd_ev`, `mu_h_fd_ev`, `carrier_density_fd_cm3`
  - uncertainty components and totals
  - power-balance columns:
    - `absorbed_power_w_cm2`
    - `radiative_power_w_cm2`
    - `nonradiative_power_w_cm2`
    - `recombination_power_w_cm2`
    - `thermalized_power_w_cm2`
    - `thermalized_power_w_cm3`
    - `thermalized_power_per_carrier_ev_s`
    - fractions and closure diagnostics
- `all_spectra_logscale.png`
- `parameters_vs_intensity.png`
- `thermalized_power_diagnostics.png`
- `pth_nT_comparison.png`
- `pth_experiment_vs_tsai.csv` (only if `TSAI_MODEL_TABLE_CSV` is provided)
- `fits/fit_spectrum_XX.png` diagnostics per spectrum

## 8) Run

Requirements:
- Python 3.10+
- `numpy`, `pandas`, `matplotlib`

Example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas matplotlib
.\.venv\Scripts\python.exe main.py
```

## 9) Main Assumptions and Limits

- High-energy approximation for GPL tail is valid in selected window.
- `A(E)` treated as approximately constant in that high-energy fit window.
- MB and FD are both evaluated with a parabolic-band DOS and fixed effective masses (`M_E_EFF`, `M_H_EFF`).
- FD currently improves degeneracy handling for carrier statistics, but uncertainty propagation is MB-based.
- Power model uses your specified recombination channel energies:
  `E_g + 3k_B T` (non-rad), `E_g + k_B T` (rad).
- Power is reported both per area (`W cm^-2`) and per volume (`W cm^-3`) using a uniform active thickness.
- Tsai parity currently uses nearest-point lookup from the supplied model table (not a full PDE/transport re-solve inside this script).

## 10) Suggested Next Improvements

1. Propagate uncertainties for FD-derived `mu_e`, `mu_h`, and `n` (parallel to MB uncertainty terms).
2. Use measured `A(E)` spectral dependence directly in fit (instead of constant `A0` in window).
3. Add optional weighted/robust regression for low-SNR high-energy tails.
4. Replace nearest-point Tsai matching by smooth interpolation on `(log10 n, T)` plus uncertainty on model predictions.
5. Introduce Monte Carlo uncertainty propagation for the full power-balance and experiment-vs-theory ratio.
6. Add CLI/config-file interface for experiment-specific runs (laser wavelength, PLQY, thickness, Tsai table path, etc.).
7. Add automated consistency checks/alerts when `P_thermalized < 0`, closure errors drift, or fitted windows become unstable.
