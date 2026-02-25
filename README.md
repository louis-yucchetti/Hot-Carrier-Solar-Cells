# Hot-Carrier Parameter Extraction from GaAs CWPL Spectra

This repository extracts hot-carrier thermodynamic and energetic quantities from calibrated GaAs photoluminescence (PL) spectra measured under different continuous-wave excitation intensities.

The analysis pipeline estimates:
- carrier temperature `T` and quasi-Fermi level splitting `Delta_mu` from the high-energy PL tail,
- electron/hole chemical potentials and carrier density using both Maxwell-Boltzmann (MB) and Fermi-Dirac (FD) statistics,
- absorbed, recombination, and thermalized power channels via detailed-balance-based bookkeeping,
- comparison-ready experimental points on `P_th(n, T)` for theory benchmarking (including optional Tsai-table overlay).

## 1) Physical Context

Under strong optical pumping, carriers can remain hotter than the lattice before fully cooling. The key observables are therefore not only carrier density `n`, but also carrier temperature `T` and how much absorbed optical power is dissipated as thermalization.

In direct-gap GaAs, the high-energy tail of PL is especially informative: in this regime, the generalized Planck law simplifies to an exponential form whose slope is controlled by `T`.

## 2) Repository Structure (Modular Refactor)

The original monolithic script has been split into focused modules:

- `main.py`: thin entrypoint.
- `hot_carrier/config.py`: all user-facing settings and physical constants.
- `hot_carrier/models.py`: result dataclasses and fit-result assembly helpers.
- `hot_carrier/analysis.py`: spectroscopy transforms, MB/FD carrier back-calculation, window search, regression, uncertainty propagation, and power-balance table generation.
- `hot_carrier/plotting.py`: publication-style plotting and optional Tsai comparison utilities.
- `hot_carrier/pipeline.py`: orchestration (validation, full run, file output, run summary).

This separation keeps numerical physics, plotting, and orchestration independent and easier to maintain.

## 3) Core Equations and Modeling

### 3.1 GPL Tail Linearization

Generalized Planck law in steady state:

`I_PL(E) ~ [2 E^2 / (h^3 c^2)] * A(E) * {exp[(E - Delta_mu)/(k_B T)] - 1}^-1`

In the high-energy tail (`E > Delta_mu + k_B T`) and approximately constant absorptivity `A(E) ~ A0` over the fit window:

`I_PL(E) ~ [2 E^2 / (h^3 c^2)] * A0 * exp[-(E - Delta_mu)/(k_B T)]`

Define a linearized ordinate:

`y(E) = ln[(h^3 c^2 / (2 E^2)) * I_PL(E)] = m E + b`

Then:
- `m = -1/(k_B T)` -> `T = -1/(k_B m)`
- intercept yields `Delta_mu_eff` (effective QFLS including unknown `ln(A0)` offset)
- corrected `Delta_mu` is obtained with configured nominal `A0`.

### 3.2 Carrier Statistics Back-Calculation

Given fitted `T` and `Delta_mu`, the code computes `mu_e`, `mu_h`, and `n` via MB and FD models.

Effective density of states:

`N_c(T) = 2 * [(m_e* k_B T)/(2 pi hbar^2)]^(3/2)`

`N_v(T) = 2 * [(m_h* k_B T)/(2 pi hbar^2)]^(3/2)`

MB (analytic, with neutrality `n=p`):

`mu_e - mu_h = k_B T ln(N_c/N_v)`

`mu_e = 0.5 * [Delta_mu - k_B T ln(N_c/N_v)]`

`mu_h = 0.5 * [Delta_mu + k_B T ln(N_c/N_v)]`

`n = N_c * exp[(mu_e - E_g/2)/(k_B T)]`

FD (numerical):

`n = N_c * F_{1/2}(eta_e)`

`p = N_v * F_{1/2}(eta_h)`

`eta_e = (mu_e - E_g/2)/(k_B T)`

`eta_h = (mu_h - E_g/2)/(k_B T)`

with `Delta_mu = mu_e + mu_h`; neutrality is solved by bisection.

Interpretation: MB and FD only change the back-calculated `(mu_e, mu_h, n)`. Fitted `T` and `Delta_mu` are unchanged because they come from the same optical tail fit.

### 3.3 Power-Balance Model

Carrier flux balance:

`phi_abs = phi_rad + phi_nonrad`

Energy balance:

`P_abs = P_th + P_rad + P_nonrad`

Absorbed laser power:

`P_abs = A_laser * P_exc`

`phi_abs = P_abs / E_laser`

Using PLQY `eta = phi_rad / (phi_rad + phi_nonrad)`:

`phi_rad = eta * phi_abs`

`phi_nonrad = (1 - eta) * phi_abs`

Per-event channel energies in the implemented model:

`E_nonrad = E_g + 3 k_B T`

`E_rad = E_g + k_B T`

Hence:

`P_rec = phi_nonrad * E_nonrad + phi_rad * E_rad`

`P_th = P_abs - P_rec`

Area to volume conversion uses active-layer thickness `d`:

`P_vol = P_area / d_cm`, with `d_cm = d_nm * 1e-7`.

## 4) Uncertainty Treatment

For each reported fit parameter `p in {T, Delta_mu_eff, Delta_mu, mu_e, mu_h, n}`, the code stores three uncertainty components and one combined value:

`sigma_total(p) = sqrt( sigma_chi2(p)^2 + sigma_range(p)^2 + sigma_A0(p)^2 )`

In implementation, this quadrature sum is applied over all finite components (NaNs are ignored).

### 4.1 `chi2` component (line-fit covariance propagation)

The GPL-tail regression is:

`y_i = m x_i + b`, with `x_i = E_i (in J)`

The residual variance estimate is:

`s^2 = RSS / (N - 2)`, where `RSS = sum_i (y_i - yhat_i)^2`

For a 2-parameter linear fit, the slope/intercept covariance is:

`Var(m)   = s^2 / Sxx`

`Var(b)   = s^2 * (1/N + xbar^2 / Sxx)`

`Cov(m,b) = -xbar * s^2 / Sxx`

with `Sxx = sum_i (x_i - xbar)^2`.

Primary parameter transforms are:

`T = -1 / (k_B m)`

`Delta_mu_eff = -b / (m e)`  (equivalent to `b k_B T / e`)

`Delta_mu = Delta_mu_eff - (k_B T / e) ln(A0)`

Using Jacobian propagation from `(m,b)` to `(T, Delta_mu_eff, Delta_mu)`:

`Cov_primary = J_primary * Cov_(m,b) * J_primary^T`

where the code uses:

`dT/dm = 1 / (k_B m^2)`

`dDelta_mu_eff/dm = b / (m^2 e)`, `dDelta_mu_eff/db = -1 / (m e)`

`dDelta_mu/dm = dDelta_mu_eff/dm - [(k_B/e) ln(A0)] * dT/dm`

`dDelta_mu/db = dDelta_mu_eff/db`

For MB-derived `(mu_e, mu_h, n)`, the code computes a numerical Jacobian
`J_MB = d(mu_e,mu_h,n)/d(T,Delta_mu)` by central finite differences, then:

`Cov_MB = J_MB * Cov_(T,Delta_mu) * J_MB^T`

and uncertainties are `sqrt` of diagonal terms.

### 4.2 `range` component (fit-window-selection uncertainty)

This term quantifies sensitivity to the chosen fit energy interval.

1. Build candidate windows:
- Use contiguous sub-windows within the scan domain.
- Keep only windows with:
  `n_points >= FIT_RANGE_SCAN_MIN_POINTS`,
  negative slope,
  `R^2 >= FIT_RANGE_SCAN_MIN_R2`,
  finite derived parameters,
  and (optionally) physical `T` bounds.

2. Score each window with AICc:

`AIC  = N ln(RSS/N) + 2k`

`AICc = AIC + [2k(k+1)] / (N-k-1)`

with `k = 2` (slope, intercept). Lower `AICc` is better. Absolute AICc has no standalone meaning; differences between windows matter.

3. Convert to Akaike weights:

`Delta_i = AICc_i - min_j(AICc_j)`

`w_i = exp(-0.5 Delta_i) / sum_j exp(-0.5 Delta_j)`

Interpretation: `w_i` is the relative support for window `i` inside the tested set.

4. Compute weighted RMS spread around the base-fit value:

`sigma_range(p) = sqrt( sum_i w_i * (p_i - p_base)^2 )`

`p_base` is the parameter from the primary selected window (auto-selected best window, or user-fixed window). So this is a "window-choice sensitivity" metric, not a posterior over `p`.

### 4.3 `A0` component (absorptivity prior uncertainty)

Nominal high-energy absorptivity is:

`A0 = 0.5 * (A0_min + A0_max)`

Its sigma is configured as either:

`sigma_A0 = (A0_max - A0_min)/sqrt(12)` (uniform model), or

`sigma_A0 = 0.5 * (A0_max - A0_min)` (half-range model).

Since `T` and `Delta_mu_eff` do not depend on `A0`, only `Delta_mu` shifts:

`dDelta_mu/dA0 = -(k_B T)/(e A0)`

`sigma_A0(Delta_mu) = |dDelta_mu/dA0| * sigma_A0`

Then `(mu_e, mu_h, n)` receive this uncertainty through the same MB Jacobian with:

`Cov_(T,Delta_mu)^(A0) = [[0,0],[0,sigma_A0(Delta_mu)^2]]`

### 4.4 Power-balance uncertainty propagation

For `P_rec` and `P_th`, first-order propagation is applied with independent inputs
`A_laser`, `eta`, and fitted `T`:

`sigma_f^2 = (df/dA_laser * sigma_A_laser)^2 + (df/deta * sigma_eta)^2 + (df/dT * sigma_T)^2`

using analytic derivatives implemented in `compute_power_balance_table()`.

Example compact forms used by the code:

`P_rec = (A_laser P_exc / E_laser) * [E_g + (3 - 2 eta) k_B T]`

`P_th  = A_laser P_exc - P_rec`

## 5) Execution Pipeline

`hot_carrier.pipeline.main()` performs:

1. Configuration validation.
2. Data load (`GaAs bulk_PL_avg_circle_4pixs.txt`) and energy sorting.
3. Raw-spectrum plotting.
4. Per-spectrum window selection and GPL-tail fit.
5. MB/FD carrier-state reconstruction.
6. Uncertainty aggregation.
7. Power-balance table augmentation.
8. Summary and diagnostic figure generation.
9. Optional Tsai-table overlay/parity export.

## 6) Inputs You Should Calibrate Experimentally

The following are currently placeholders/defaults and should be set from experiment:
- `ABSORPTIVITY_AT_LASER`
- `ABSORPTIVITY_AT_LASER_SIGMA`
- `PLQY_ETA`
- `PLQY_ETA_SIGMA`
- verify `LASER_WAVELENGTH_NM`
- verify `ACTIVE_LAYER_THICKNESS_NM`

Optional model overlay input:
- `TSAI_MODEL_TABLE_CSV` (columns: `n_cm3`, `temperature_k`, `p_th_w_cm3`)

## 7) How to Run

Using the project virtual environment:

```powershell
.\.venv\Scripts\python.exe main.py
```

## 8) Outputs

Generated in `outputs/`:

- `fit_results.csv`
- `all_spectra_logscale.png`
- `parameters_vs_intensity.png`
- `thermalized_power_diagnostics.png`
- `pth_nT_comparison.png`
- per-spectrum fits in `outputs/fits/`
- optional `pth_experiment_vs_tsai.csv`

`fit_results.csv` includes 4 groups of fields:
- fit-state: `temperature_k`, `qfls_ev`, `qfls_effective_ev`, MB/FD carrier quantities.
- uncertainty terms: chi2/range/A0/total for key parameters.
- power-balance quantities: fluxes, powers (both `W cm^-2` and `W cm^-3`), closure diagnostics.
- normalized diagnostics: thermalized/recombination fractions and per-carrier thermalization rate.

## 9) Tsai-Model Comparison Workflow

1. Run this pipeline to generate experimental `(n, T, P_th)` points.
2. Prepare a Tsai-model CSV with columns `n_cm3`, `temperature_k`, `p_th_w_cm3`.
3. Set `TSAI_MODEL_TABLE_CSV` in `hot_carrier/config.py`.
4. Re-run; inspect:
   - manifold view (experimental points + model contours),
   - direct parity-style comparison at measured `(n, T)`,
   - `outputs/pth_experiment_vs_tsai.csv` ratios.

## 10) Notes on Scope

- FD values are reported from the fitted `T` and `Delta_mu`; separate FD-specific uncertainty propagation is not yet implemented.
- The model is intentionally compact and comparison-oriented; it is a thermodynamic reconstruction framework, not a full microscopic transport solver.
