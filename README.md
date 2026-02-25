# Hot-Carrier Parameter Extraction from Continuous-Wave Photoluminescence (CWPL) Spectra of Gallium Arsenide (GaAs)

This repository extracts hot-carrier thermodynamic and energy-flow quantities from calibrated photoluminescence (PL) spectra measured on GaAs under continuous-wave excitation. The workflow is meant to be transparent: each reported quantity can be traced back to a specific equation and code path.

For each spectrum, the pipeline reports:
- carrier temperature `T`,
- quasi-Fermi level splitting (QFLS), written as `Delta_mu`,
- electron and hole chemical potentials plus carrier density from Maxwell-Boltzmann (MB) and Fermi-Dirac (FD) statistics,
- absorbed, recombination, and thermalized power channels,
- uncertainty components from line-fit statistics, fit-window choice, and absorptivity prior.

## 1) Why CWPL on GaAs

### 1.1 Physical motivation

Under strong optical pumping, carriers can stay hotter than the lattice before they fully cool. If that happens, carrier density alone is not enough; the carrier temperature also matters. PL is useful here because the emitted photon energies carry information about the carrier distribution.

GaAs is a direct-bandgap semiconductor, so radiative recombination is strong near the band edge. That makes the high-energy side of the PL spectrum a practical temperature probe. Continuous-wave (CW) excitation is used so the system reaches steady state, which lets generation, recombination, and cooling be analyzed with balance equations.

### 1.2 Why the high-energy tail is fitted

In the high-energy tail, the generalized Planck law (GPL) becomes an exponential in energy. The exponential slope is set by `T`, so fitting that region gives a direct temperature estimate. The intercept then gives `Delta_mu` after correction for absorptivity.

## 2) Experimental Data Represented in This Repository

The dataset is expected in `GaAs bulk_PL_avg_circle_4pixs.txt`:
- one energy axis in electron-volts (eV),
- one PL intensity column per excitation condition.

Excitation intensities are provided in `EXCITATION_INTENSITY_W_CM2` in `hot_carrier/config.py`.

Each spectrum is fitted independently, then all extracted values are collected into one results table for trend analysis versus excitation intensity.

## 3) Emission Model and Parameter Extraction

### 3.1 Generalized Planck law (GPL)

PL intensity is modeled as:

`I_PL(E) ~ [2 E^2 / (h^3 c^2)] * A(E) * {exp[(E - Delta_mu)/(k_B T)] - 1}^-1`

where:
- `E` is photon energy,
- `A(E)` is absorptivity,
- `Delta_mu` is QFLS,
- `k_B` is Boltzmann constant,
- `h` is Planck constant,
- `c` is speed of light.

### 3.2 High-energy approximation and linearization

For `E > Delta_mu + k_B T`, and over a narrow window where `A(E) ~ A0`:

`I_PL(E) ~ [2 E^2 / (h^3 c^2)] * A0 * exp[-(E - Delta_mu)/(k_B T)]`

Define:

`y(E) = ln[(h^3 c^2 / (2 E^2)) * I_PL(E)] = m E + b`

In code, `E` is converted to joules before fitting. From slope `m` and intercept `b`:

`T = -1 / (k_B m)`

`Delta_mu_eff = -b / (m e) = (b k_B T)/e`

`Delta_mu = Delta_mu_eff - (k_B T / e) ln(A0)`

`Delta_mu_eff` is the effective QFLS before correction by nominal `A0`, and `e` is elementary charge.

### 3.3 Fit-window selection

A scan domain is built from `WINDOW_SEARCH_MIN_EV` and `WINDOW_SEARCH_MAX_EV`, with optional peak offset (`WINDOW_PEAK_OFFSET_EV`). Contiguous sub-windows are tested and filtered by:
- minimum point count,
- negative slope,
- minimum coefficient of determination (`R^2`),
- physical temperature bounds (if enabled).

Windows are scored with the Akaike Information Criterion (AIC) and corrected Akaike Information Criterion (AICc):

`AIC = N ln(RSS/N) + 2k`

`AICc = AIC + [2k(k+1)] / (N-k-1)`

where `N` is number of points, `k=2` for slope/intercept, and `RSS` is residual sum of squares. The window with minimum AICc is used as the primary fit window.

## 4) Carrier-State Reconstruction from `T` and `Delta_mu`

### 4.1 Effective density of states

For parabolic bands:

`N_c(T) = 2 * [(m_e* k_B T)/(2 pi hbar^2)]^(3/2)`

`N_v(T) = 2 * [(m_h* k_B T)/(2 pi hbar^2)]^(3/2)`

with electron and hole effective masses `m_e*`, `m_h*`.

### 4.2 Maxwell-Boltzmann (MB) model

Using charge neutrality `n = p` and `Delta_mu = mu_e + mu_h`:

`mu_e - mu_h = (k_B T / e) ln(N_c/N_v)`

`mu_e = 0.5 * [Delta_mu - (k_B T / e) ln(N_c/N_v)]`

`mu_h = 0.5 * [Delta_mu + (k_B T / e) ln(N_c/N_v)]`

`n = N_c * exp[((mu_e - E_g/2) e)/(k_B T)]`

### 4.3 Fermi-Dirac (FD) model

The complete FD integral of order one-half is used:

`F_{1/2}(eta) = (2/sqrt(pi)) * integral_0^inf sqrt(eps)/(1 + exp(eps-eta)) d eps`

with:

`n = N_c * F_{1/2}(eta_e)`

`p = N_v * F_{1/2}(eta_h)`

`eta_e = ((mu_e - E_g/2) e)/(k_B T)`

`eta_h = ((mu_h - E_g/2) e)/(k_B T)`

and `Delta_mu = mu_e + mu_h`. Neutrality is solved numerically by bisection.

The optical fit provides `T` and `Delta_mu`; MB and FD only change the back-calculated `mu_e`, `mu_h`, and `n`.

## 5) Power-Balance Model

### 5.1 Flux and power relations

Carrier flux balance:

`phi_abs = phi_rad + phi_nonrad`

Power balance:

`P_abs = P_th + P_rad + P_nonrad`

with:

`P_abs = A_laser * P_exc`

`phi_abs = P_abs / E_laser`

`A_laser` is absorptivity at the laser wavelength, `P_exc` is incident power density, and `E_laser` is laser photon energy.

Using photoluminescence quantum yield (PLQY), `eta`:

`phi_rad = eta * phi_abs`

`phi_nonrad = (1 - eta) * phi_abs`

### 5.2 Channel energies and thermalized power

The implemented channel energies are:

`E_nonrad = E_g + 3 k_B T`

`E_rad = E_g + k_B T`

So:

`P_rec = phi_nonrad * E_nonrad + phi_rad * E_rad`

`P_th = P_abs - P_rec`

Area powers are converted to volume powers with active-layer thickness `d`:

`P_vol = P_area / d_cm`, with `d_cm = d_nm * 1e-7`.

## 6) Uncertainty Treatment

For each fitted parameter `p in {T, Delta_mu_eff, Delta_mu, mu_e, mu_h, n}`:

`sigma_total(p) = sqrt( sigma_chi2(p)^2 + sigma_range(p)^2 + sigma_A0(p)^2 )`

The code combines finite components in quadrature.

### 6.1 Chi-squared (`chi2`) component

For `y_i = m x_i + b`, residual variance is:

`s^2 = RSS / (N - 2)`

with `Sxx = sum_i (x_i - xbar)^2` and:

`Var(m)   = s^2 / Sxx`

`Var(b)   = s^2 * (1/N + xbar^2 / Sxx)`

`Cov(m,b) = -xbar * s^2 / Sxx`

A Jacobian transform from `(m,b)` to `(T, Delta_mu_eff, Delta_mu)` is used:

`Cov_primary = J_primary * Cov_(m,b) * J_primary^T`

with:

`dT/dm = 1 / (k_B m^2)`

`dDelta_mu_eff/dm = b / (m^2 e)`, `dDelta_mu_eff/db = -1/(m e)`

`dDelta_mu/dm = dDelta_mu_eff/dm - [(k_B/e) ln(A0)] * dT/dm`

`dDelta_mu/db = dDelta_mu_eff/db`

For MB-derived `(mu_e, mu_h, n)`, a numerical Jacobian
`J_MB = d(mu_e,mu_h,n)/d(T,Delta_mu)` is computed by central differences:

`Cov_MB = J_MB * Cov_(T,Delta_mu) * J_MB^T`

Uncertainties are the square roots of diagonal variances.

### 6.2 Fit-range (`range`) component

This term captures sensitivity to fit-window choice.

1. Candidate windows are enumerated over the scan domain.
2. Windows are filtered by slope, `R^2`, point count, and physical bounds.
3. AICc is computed for each remaining window.
4. Relative weights are formed:

`Delta_i = AICc_i - min_j(AICc_j)`

`w_i = exp(-0.5 Delta_i) / sum_j exp(-0.5 Delta_j)`

5. A weighted root-mean-square (RMS) spread around the base-fit value is used:

`sigma_range(p) = sqrt( sum_i w_i * (p_i - p_base)^2 )`

This is a window-sensitivity metric, not a full posterior over parameters.

### 6.3 Absorptivity (`A0`) component

Nominal `A0`:

`A0 = 0.5 * (A0_min + A0_max)`

Configured uncertainty options:

`sigma_A0 = (A0_max - A0_min)/sqrt(12)` (uniform model)

or

`sigma_A0 = 0.5 * (A0_max - A0_min)` (half-range model)

Direct effect on `Delta_mu`:

`dDelta_mu/dA0 = -(k_B T)/(e A0)`

`sigma_A0(Delta_mu) = |dDelta_mu/dA0| * sigma_A0`

Propagation to MB-derived quantities again uses `J_MB`.

### 6.4 Power-balance uncertainty propagation

For `f in {P_rec, P_th}`, first-order propagation with independent inputs is used:

`sigma_f^2 = (df/dA_laser * sigma_A_laser)^2 + (df/deta * sigma_eta)^2 + (df/dT * sigma_T)^2`

Analytic derivatives are implemented in `compute_power_balance_table()`.

## 7) Processing Pipeline

`hot_carrier.pipeline.main()` runs the following sequence:

1. Validate configuration.
2. Load spectra and sort by energy.
3. Plot raw spectra.
4. Select a fit window per spectrum.
5. Fit GPL tail.
6. Extract `T`, `Delta_mu_eff`, and `Delta_mu`.
7. Reconstruct MB and FD carrier quantities.
8. Compute and combine uncertainty components.
9. Compute power-balance quantities and uncertainties.
10. Export tables and diagnostic figures.
11. Optionally generate Tsai-model comparison outputs.

## 8) Repository Structure

- `main.py`: entry point.
- `hot_carrier/config.py`: settings and constants.
- `hot_carrier/models.py`: data classes and result assembly.
- `hot_carrier/analysis.py`: fitting, uncertainty propagation, and power-balance calculations.
- `hot_carrier/plotting.py`: figure generation and comparison plots.
- `hot_carrier/pipeline.py`: end-to-end orchestration.

## 9) Inputs to Calibrate Experimentally

These values should be set from your experiment:
- `ABSORPTIVITY_AT_LASER`
- `ABSORPTIVITY_AT_LASER_SIGMA`
- `PLQY_ETA`
- `PLQY_ETA_SIGMA`
- `LASER_WAVELENGTH_NM`
- `ACTIVE_LAYER_THICKNESS_NM`

For optional Tsai comparison, set `TSAI_MODEL_TABLE_CSV` to a comma-separated values (CSV) file with columns:
- `n_cm3`
- `temperature_k`
- `p_th_w_cm3`

## 10) Run

```powershell
.\.venv\Scripts\python.exe main.py
```

## 11) Outputs

Files written to `outputs/`:
- `fit_results.csv`
- `all_spectra_logscale.png`
- `parameters_vs_intensity.png`
- `thermalized_power_diagnostics.png`
- `pth_nT_comparison.png`
- per-spectrum fit plots in `outputs/fits/`
- optional `pth_experiment_vs_tsai.csv`

`fit_results.csv` includes:
- fitted state variables (`temperature_k`, `qfls_effective_ev`, `qfls_ev`, MB and FD carrier quantities),
- uncertainty terms (`*_err_chi2_*`, `*_err_range_*`, `*_err_a0_*`, `*_err_total_*`),
- flux and power channels in `W cm^-2` and `W cm^-3`,
- closure diagnostics and normalized partition metrics.

## 12) Optional Tsai-Model Comparison Workflow

1. Run this pipeline to generate experimental `(n, T, P_th)` points.
2. Prepare a CSV file with columns `n_cm3`, `temperature_k`, `p_th_w_cm3`.
3. Set `TSAI_MODEL_TABLE_CSV` in `hot_carrier/config.py`.
4. Re-run the pipeline.
5. Inspect the comparison plots and `outputs/pth_experiment_vs_tsai.csv`.

## 13) Current Limitations

- FD values are reported, but FD-specific uncertainty propagation is not yet implemented.
- The model is a compact thermodynamic reconstruction, not a full microscopic transport solver.
