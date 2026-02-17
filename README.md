# Hot Carrier Properties Extraction from GaAs PL Spectra

This project fits experimental photoluminescence (PL) spectra to a simplified form of the Generalized Planck Law (GPL) in the high-energy regime.  
From each spectrum, the code extracts:

- carrier temperature `T`
- quasi-Fermi level splitting `Delta mu` (QFLS)
- optional derived quantities under Maxwell-Boltzmann assumptions: `mu_e`, `mu_h`, and carrier density `n`

It also generates publication-style figures for quality control and trend analysis versus excitation intensity.

## 1) Scientific objective

You have a set of PL spectra of a GaAs sample measured at different excitation intensities.  
Each spectrum is a photon-energy distribution and contains information about:

- how hot the carrier population is (`T`)
- how far the electron-hole system is from equilibrium (`Delta mu`)

The practical workflow is:

1. visualize all spectra
2. identify a high-energy domain where the linearized GPL is valid
3. fit each spectrum in that domain
4. compare extracted parameters across excitation intensity

## 2) Physics background (intuitive)

In steady-state PL, emitted intensity is governed by the generalized Planck law:

`I_PL(E) ~ [2 E^2 / (h^3 c^2)] * A(E) * {exp[(E - Delta mu)/(k_B T)] - 1}^-1`

For sufficiently high energies (`E > Delta mu + k_B T`) and if absorptivity is approximately constant (`A(E) ~= A0`) over the fit window:

`I_PL(E) ~ [2 E^2 / (h^3 c^2)] * A0 * exp[-(E - Delta mu)/(k_B T)]`

Taking the logarithm of a transformed intensity gives a linear relation:

`ln[(h^3 c^2 / 2 E^2) I_PL] = -(E - Delta mu)/(k_B T) + ln(A0)`

So in the correct high-energy domain:

- slope gives temperature (`slope = -1/(k_B T)`)
- intercept gives `Delta mu` up to the unknown `ln(A0)` offset

Important consequence:

- if `A0` is not independently calibrated, absolute `Delta mu` is an "effective" value
- temperature extraction from slope is much less sensitive to that offset

## 3) What the code does end-to-end

Main script: `main.py`

1. **Load spectra**
   - reads `GaAs bulk_PL_avg_circle_4pixs.txt` with `sep=';'` and energy index in eV
2. **Sort energies**
   - data is sorted in increasing energy for fitting/plotting consistency
3. **Plot all raw spectra**
   - log-scale intensity plot for quick visual inspection
4. **Select fit window per spectrum (automatic)**
   - find spectral peak
   - search windows above `E_peak + offset` in a configurable energy range
   - test many contiguous candidate windows
   - keep physically valid linear segments (negative slope, good `R^2`, realistic `T`)
   - choose best window using a score (linearity + window length + slight high-energy preference)
5. **Fit linearized GPL in selected window**
   - linear regression on transformed signal
   - extract `T`, `Delta mu_eff`, `Delta mu`
6. **Compute optional derived quantities**
   - from GaAs effective masses and MB expressions: `mu_e`, `mu_h`, `n`
7. **Generate diagnostic figure per spectrum**
   - top panel: raw spectrum + fitted model + selected window
   - bottom panel: linearized data + regression line
8. **Aggregate over all intensities**
   - create trend figure: `T`, `Delta mu`, `mu_e/mu_h`, `n` vs excitation
9. **Export all artifacts**
   - CSV table with fitted parameters
   - PNG and PDF figures (publication-friendly style)

## 4) Repository structure

- `main.py`: full analysis pipeline
- `GaAs bulk_PL_avg_circle_4pixs.txt`: PL dataset (22 spectra)
- `outputs/`: generated results
  - `all_spectra_logscale.png/.pdf`
  - `parameters_vs_intensity.png/.pdf`
  - `fit_results.csv`
  - `fits/fit_spectrum_XX.png/.pdf` for each spectrum

## 5) Setup and run

## Requirements

- Python 3.10+ (tested in project `.venv`)
- Packages:
  - `numpy`
  - `pandas`
  - `matplotlib`

Install (example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas matplotlib
```

Run:

```powershell
.\.venv\Scripts\python.exe main.py
```

## 6) Key configuration parameters

At the top of `main.py`:

- `AUTO_SELECT_FIT_WINDOW`
  - `True`: automatic per-spectrum window selection
  - `False`: use fixed `[FIT_ENERGY_MIN_EV, FIT_ENERGY_MAX_EV]`
- `WINDOW_SEARCH_MIN_EV`, `WINDOW_SEARCH_MAX_EV`
  - global search bounds for auto selection
- `WINDOW_PEAK_OFFSET_EV`
  - start searching above `E_peak + offset` to target high-energy tail
- `WINDOW_MIN_POINTS`
  - minimum points in a candidate window
- `WINDOW_MIN_R2`
  - minimum linearity requirement
- `WINDOW_T_MIN_K`, `WINDOW_T_MAX_K`
  - physical bounds to reject unphysical fits
- `ASSUMED_A0`
  - absorptivity prefactor for converting effective QFLS to absolute QFLS
- `EG_EV`, `M_E_EFF`, `M_H_EFF`
  - GaAs parameters used in MB-based `mu_e`, `mu_h`, `n` derivation
- `SAVE_DPI`, `EXPORT_PDF`
  - figure quality/output format controls

## 7) Understanding `fit_results.csv`

Main columns:

- `spectrum_id`: source spectrum column index
- `intensity_w_cm2`: excitation intensity associated with spectrum
- `fit_min_ev`, `fit_max_ev`: selected fit window bounds for that spectrum
- `window_mode`: whether auto/fallback path was used
- `n_points_fit`: number of data points used in regression
- `r2`: fit quality in linearized space
- `temperature_k`: extracted carrier temperature
- `qfls_effective_ev`: effective `Delta mu` from intercept (includes unknown `A0`)
- `qfls_ev`: corrected QFLS using `ASSUMED_A0`
- `mu_e_ev`, `mu_h_ev`, `carrier_density_cm3`: MB-derived quantities

## 8) Quality checks to always perform

Before trusting trends, inspect:

1. each `outputs/fits/fit_spectrum_XX.*` plot
2. selected windows (`fit_min_ev`, `fit_max_ev`) for consistency across spectra
3. `r2` values and outliers
4. physical plausibility of `T` and `Delta mu` trends vs excitation

## 9) Assumptions and limitations

- High-energy approximation is used (`E > Delta mu + k_B T` behavior).
- `A(E)` is treated as constant in the chosen window.
- Without independent optical calibration of `A0`, absolute `Delta mu` may be shifted.
- `mu_e`, `mu_h`, `n` are currently MB-based approximations; at high density/degeneracy, a full Fermi-Dirac treatment is preferable.
- Results depend on good spectral SNR and robust window selection.

## 10) Suggested next improvements

- replace MB carrier extraction with full Fermi-Dirac integrals + charge neutrality solver
- propagate fit uncertainty to confidence intervals on `T`, `Delta mu`, and `n`
- add robust regression options (e.g., weighted fits in low-SNR tails)
- add batch support for multiple input files and automated report generation

---

If you want this README adapted to a specific journal workflow (methods section style, SI-ready equations, figure naming convention), that can be added directly.
