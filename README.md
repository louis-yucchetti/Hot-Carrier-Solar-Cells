# Hot-Carrier Parameter Extraction from Continuous-Wave Photoluminescence (CWPL) Spectra of Gallium Arsenide (GaAs)

Thermodynamic and energetic hot-carrier quantities are extracted in this repository from calibrated photoluminescence (PL) spectra measured under multiple continuous-wave optical excitation intensities. A compact but equation-explicit workflow is implemented so that extracted quantities can be compared directly against theory.

The following outputs are produced for each spectrum:
- carrier temperature `T` and quasi-Fermi level splitting `Delta_mu`,
- electron and hole chemical potentials and carrier density from Maxwell-Boltzmann (MB) and Fermi-Dirac (FD) statistics,
- absorbed, recombination, and thermalized power channels,
- uncertainty terms from line-fit statistics, fit-window choice, and absorptivity prior.

## 1) Why Continuous-Wave Photoluminescence Is Used for Hot Carriers

### 1.1 Physical motivation

Under strong optical pumping, carrier temperatures can be driven above lattice temperature for a finite time before full cooling is completed. In that regime, information about carrier temperature is encoded in the energy distribution of emitted photons, not only in the total carrier population.

In direct-bandgap GaAs, radiative recombination is strong near the band edge. Because PL is emitted directly by electron-hole recombination, the high-energy side of the PL spectrum can be used as a thermometer for the carrier ensemble. A continuous-wave (CW) excitation condition is used so that a steady-state balance between generation, recombination, and cooling can be analyzed.

### 1.2 Why the high-energy tail is informative

An exponential dependence in the high-energy PL tail is predicted by the generalized Planck law (GPL). When an exponential tail is observed, its slope is controlled by `T`. The high-energy tail is therefore fitted in a physically selected energy window, and the resulting slope and intercept are converted into thermodynamic parameters.

## 2) What Experiment Is Represented

A set of steady-state PL spectra is assumed to be measured from a GaAs sample while the excitation intensity is stepped across a known list of incident powers (stored in `EXCITATION_INTENSITY_W_CM2` in `hot_carrier/config.py`).

The file `GaAs bulk_PL_avg_circle_4pixs.txt` is expected to contain:
- an energy axis in electron-volts (eV),
- one PL intensity column per excitation condition.

Each spectrum is processed independently, then all extracted parameters are assembled into one table for trend analysis versus excitation intensity.

## 3) Core Emission Model and Parameter Extraction

### 3.1 Generalized Planck law (GPL)

In steady state, the PL spectral intensity is modeled as:

`I_PL(E) ~ [2 E^2 / (h^3 c^2)] * A(E) * {exp[(E - Delta_mu)/(k_B T)] - 1}^-1`

where:
- `E` is photon energy,
- `A(E)` is absorptivity,
- `Delta_mu` is quasi-Fermi level splitting (QFLS),
- `k_B` is Boltzmann constant,
- `h` is Planck constant,
- `c` is speed of light.

### 3.2 High-energy approximation and linearization

In the tail regime (`E > Delta_mu + k_B T`) and over a narrow fitting interval where `A(E)` is approximated as a constant `A0`, the model becomes:

`I_PL(E) ~ [2 E^2 / (h^3 c^2)] * A0 * exp[-(E - Delta_mu)/(k_B T)]`

A linearized ordinate is then defined:

`y(E) = ln[(h^3 c^2 / (2 E^2)) * I_PL(E)] = m E + b`

In the implementation, `E` is converted to joules during fitting (`x = E_J`). From the fitted slope `m` and intercept `b`:

`T = -1 / (k_B m)`

`Delta_mu_eff = -b / (m e) = (b k_B T)/e`

`Delta_mu = Delta_mu_eff - (k_B T / e) ln(A0)`

where `e` is elementary charge and `Delta_mu_eff` is the effective QFLS before correction by nominal `A0`.

### 3.3 Automatic fit-window selection

A candidate scan domain is built from physically meaningful limits (`WINDOW_SEARCH_MIN_EV`, `WINDOW_SEARCH_MAX_EV`) and, when enabled, an offset above the PL peak (`WINDOW_PEAK_OFFSET_EV`).

All contiguous sub-windows satisfying minimum sample count and fit quality are evaluated. For each window:
- a linear fit is performed,
- slope sign, coefficient of determination (`R^2`), and physical `T` bounds are checked,
- corrected Akaike Information Criterion (AICc) is computed.

The Akaike Information Criterion (AIC) and corrected Akaike Information Criterion (AICc) used by the code are:

`AIC = N ln(RSS/N) + 2k`

`AICc = AIC + [2k(k+1)] / (N-k-1)`

where:
- `N` is number of points in the window,
- `k = 2` for slope and intercept,
- `RSS` is residual sum of squares.

The window with minimum AICc is selected for the primary fit. Lower AICc indicates stronger support among candidate windows after finite-sample correction.

## 4) Carrier-State Reconstruction from Fitted `T` and `Delta_mu`

### 4.1 Effective density of states

For parabolic bands:

`N_c(T) = 2 * [(m_e* k_B T)/(2 pi hbar^2)]^(3/2)`

`N_v(T) = 2 * [(m_h* k_B T)/(2 pi hbar^2)]^(3/2)`

where `m_e*` and `m_h*` are effective masses.

### 4.2 Maxwell-Boltzmann (MB) branch

Using charge neutrality (`n = p`) and `Delta_mu = mu_e + mu_h`:

`mu_e - mu_h = (k_B T / e) ln(N_c/N_v)`

`mu_e = 0.5 * [Delta_mu - (k_B T / e) ln(N_c/N_v)]`

`mu_h = 0.5 * [Delta_mu + (k_B T / e) ln(N_c/N_v)]`

`n = N_c * exp[((mu_e - E_g/2) e)/(k_B T)]`

### 4.3 Fermi-Dirac (FD) branch

The complete FD integral of order one-half is used:

`F_{1/2}(eta) = (2/sqrt(pi)) * integral_0^inf sqrt(eps)/(1 + exp(eps-eta)) d eps`

with:

`n = N_c * F_{1/2}(eta_e)`

`p = N_v * F_{1/2}(eta_h)`

`eta_e = ((mu_e - E_g/2) e)/(k_B T)`

`eta_h = ((mu_h - E_g/2) e)/(k_B T)`

and `Delta_mu = mu_e + mu_h`. Neutrality is enforced numerically by bisection.

`T` and `Delta_mu` are determined from the same optical fit, so MB and FD differ only in the back-calculated `mu_e`, `mu_h`, and `n`.

## 5) Power-Balance Model

### 5.1 Flux and power bookkeeping

Carrier flux balance is imposed as:

`phi_abs = phi_rad + phi_nonrad`

Power balance is imposed as:

`P_abs = P_th + P_rad + P_nonrad`

with:

`P_abs = A_laser * P_exc`

`phi_abs = P_abs / E_laser`

where `A_laser` is absorptivity at laser energy, `P_exc` is incident excitation power density, and `E_laser` is laser photon energy.

Using photoluminescence quantum yield (PLQY) `eta`:

`phi_rad = eta * phi_abs`

`phi_nonrad = (1 - eta) * phi_abs`

### 5.2 Channel energies and thermalized power

In the implemented compact model:

`E_nonrad = E_g + 3 k_B T`

`E_rad = E_g + k_B T`

Hence:

`P_rec = phi_nonrad * E_nonrad + phi_rad * E_rad`

`P_th = P_abs - P_rec`

Area-based powers are converted to volumetric powers by active-layer thickness `d`:

`P_vol = P_area / d_cm`, with `d_cm = d_nm * 1e-7`.

## 6) Uncertainty Treatment

For each reported parameter `p in {T, Delta_mu_eff, Delta_mu, mu_e, mu_h, n}`, three components are computed and combined:

`sigma_total(p) = sqrt( sigma_chi2(p)^2 + sigma_range(p)^2 + sigma_A0(p)^2 )`

Finite components are combined in quadrature; undefined components are ignored.

### 6.1 `chi2` (chi-squared) component: covariance propagation from line fit

For `y_i = m x_i + b`, residual variance is estimated as:

`s^2 = RSS / (N - 2)`

Let `Sxx = sum_i (x_i - xbar)^2`. The slope/intercept covariance is:

`Var(m)   = s^2 / Sxx`

`Var(b)   = s^2 * (1/N + xbar^2 / Sxx)`

`Cov(m,b) = -xbar * s^2 / Sxx`

A Jacobian map from `(m,b)` to `(T, Delta_mu_eff, Delta_mu)` is applied:

`Cov_primary = J_primary * Cov_(m,b) * J_primary^T`

with:

`dT/dm = 1 / (k_B m^2)`

`dDelta_mu_eff/dm = b / (m^2 e)`, `dDelta_mu_eff/db = -1/(m e)`

`dDelta_mu/dm = dDelta_mu_eff/dm - [(k_B/e) ln(A0)] * dT/dm`

`dDelta_mu/db = dDelta_mu_eff/db`

For MB-derived `(mu_e, mu_h, n)`, a numerical central-difference Jacobian
`J_MB = d(mu_e,mu_h,n)/d(T,Delta_mu)` is used:

`Cov_MB = J_MB * Cov_(T,Delta_mu) * J_MB^T`

Standard uncertainties are taken as square roots of diagonal variances.

### 6.2 `range` component: uncertainty from fit-window choice

This component quantifies sensitivity to plausible fit windows.

1. Candidate windows are enumerated over the scan domain.
2. Windows are filtered by point count, negative slope, `R^2`, and optional physical bounds.
3. Each window receives AICc.
4. Relative weights are computed:

`Delta_i = AICc_i - min_j(AICc_j)`

`w_i = exp(-0.5 Delta_i) / sum_j exp(-0.5 Delta_j)`

5. A weighted root-mean-square (RMS) spread around the base-fit value is computed:

`sigma_range(p) = sqrt( sum_i w_i * (p_i - p_base)^2 )`

This term should be interpreted as window-selection sensitivity, not as a full Bayesian posterior.

### 6.3 `A0` component: absorptivity prior propagation

Nominal `A0` is set to midrange:

`A0 = 0.5 * (A0_min + A0_max)`

Its uncertainty is configured as either:

`sigma_A0 = (A0_max - A0_min)/sqrt(12)` for a uniform-prior model,

or:

`sigma_A0 = 0.5 * (A0_max - A0_min)` for a half-range model.

Only `Delta_mu` is directly affected by `A0`:

`dDelta_mu/dA0 = -(k_B T)/(e A0)`

`sigma_A0(Delta_mu) = |dDelta_mu/dA0| * sigma_A0`

Propagation to MB-derived quantities is then obtained with the same Jacobian `J_MB`.

### 6.4 Power-balance uncertainty propagation

For derived functions `f in {P_rec, P_th}`, first-order independent-input propagation is applied:

`sigma_f^2 = (df/dA_laser * sigma_A_laser)^2 + (df/deta * sigma_eta)^2 + (df/dT * sigma_T)^2`

Analytic derivatives implemented in `compute_power_balance_table()` are used.

## 7) Data Processing and Analysis Workflow

The pipeline in `hot_carrier.pipeline.main()` is executed as:

1. Configuration values are validated.
2. Spectral data are loaded and energy-ordered.
3. Raw spectra are plotted.
4. A fit window is selected for each spectrum.
5. GPL-tail linear regression is performed.
6. `T`, `Delta_mu_eff`, and `Delta_mu` are extracted.
7. MB and FD carrier-state quantities are reconstructed.
8. `chi2`, `range`, and `A0` uncertainties are computed and combined.
9. Power-balance quantities and their uncertainties are appended.
10. Summary plots and tables are exported.
11. Optional Tsai-model comparison products are generated.

## 8) Repository Structure

- `main.py`: entry point.
- `hot_carrier/config.py`: settings and physical constants.
- `hot_carrier/models.py`: dataclasses and result assembly helpers.
- `hot_carrier/analysis.py`: spectral transforms, fitting, uncertainty propagation, and power-balance calculations.
- `hot_carrier/plotting.py`: figure generation and model-comparison plotting.
- `hot_carrier/pipeline.py`: orchestration of full analysis.

## 9) Inputs That Should Be Calibrated Experimentally

The following are currently user-configured and should be set from measured values:
- `ABSORPTIVITY_AT_LASER`
- `ABSORPTIVITY_AT_LASER_SIGMA`
- `PLQY_ETA`
- `PLQY_ETA_SIGMA`
- `LASER_WAVELENGTH_NM`
- `ACTIVE_LAYER_THICKNESS_NM`

For optional Tsai-model overlay, `TSAI_MODEL_TABLE_CSV` may be set to a comma-separated values (CSV) file with:
- `n_cm3`
- `temperature_k`
- `p_th_w_cm3`

## 10) How to Run

A virtual environment is assumed. The pipeline can be run with:

```powershell
.\.venv\Scripts\python.exe main.py
```

## 11) Output Files

The following files are written to `outputs/`:
- `fit_results.csv`
- `all_spectra_logscale.png`
- `parameters_vs_intensity.png`
- `thermalized_power_diagnostics.png`
- `pth_nT_comparison.png`
- per-spectrum fit plots in `outputs/fits/`
- optional `pth_experiment_vs_tsai.csv`

`fit_results.csv` contains:
- fitted state variables (`temperature_k`, `qfls_effective_ev`, `qfls_ev`, MB and FD carrier quantities),
- uncertainty components (`*_err_chi2_*`, `*_err_range_*`, `*_err_a0_*`, `*_err_total_*`),
- power-balance and flux channels in area and volume units,
- closure diagnostics and normalized energy-partition metrics.

## 12) Optional Tsai-Model Comparison Workflow

An optional comparison can be produced between experimentally extracted thermalized power and an external Tsai-model table.

The following sequence is used:

1. Experimental points `(n, T, P_th)` are generated by running this pipeline.
2. A comma-separated values (CSV) table is prepared with columns `n_cm3`, `temperature_k`, `p_th_w_cm3`.
3. `TSAI_MODEL_TABLE_CSV` is set in `hot_carrier/config.py`.
4. The pipeline is re-run.
5. Comparison graphics and `outputs/pth_experiment_vs_tsai.csv` are exported.

## 13) Scope and Current Limitations

- FD values are reported from fitted `T` and `Delta_mu`, but a separate FD-specific uncertainty propagation branch is not yet implemented.
- A compact thermodynamic reconstruction is implemented; a full microscopic transport solver is not implemented in this repository.
