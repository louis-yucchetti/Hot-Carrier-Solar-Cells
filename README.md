# Hot-Carrier Analysis from Continuous-Wave Photoluminescence in GaAs

This repository analyzes calibrated continuous-wave photoluminescence (CWPL)
spectra from bulk GaAs and extracts steady-state hot-carrier quantities from
them. The main outputs are:

- carrier temperature `T`,
- quasi-Fermi level splitting `Delta_mu`,
- electron and hole chemical potentials plus carrier density from both
  Maxwell-Boltzmann (MB) and Fermi-Dirac (FD) carrier statistics,
- a diagnostic for when MB is no longer self-consistent for the photon
  occupation part of the generalized Planck law,
- absorbed, recombination, and thermalized power densities,
- a Tsai-style electron cooling comparison based on LO-phonon scattering.

The code is written as transparent research analysis code rather than as a
black-box package. Most of the scientific logic lives in
`hot_carrier/analysis.py` and `hot_carrier/tsai_model.py`, the workflow is
orchestrated in `hot_carrier/pipeline.py`, and the user-facing settings are in
`hot_carrier/config.py`.

## 1. Project Scope

The repository currently contains one bundled GaAs dataset,
`GaAs bulk_PL_avg_circle_4pixs.txt`, one PLQY table,
`GaAs bulk_PLQY_results.csv`, several reference PDFs in `literature/`, and a
full set of generated outputs in `outputs/`.

With the current configuration and the bundled data, the project processes:

| Item | Current value |
| --- | --- |
| Number of spectra | 22 |
| Energy samples per spectrum | 141 |
| Photon-energy range | `1.265` to `1.771 eV` |
| Excitation-intensity range | `9.46e1` to `1.30e5 W cm^-2` |
| Extracted temperature range | `298.9` to `378.1 K` |
| Extracted `Delta_mu` range | `1.121` to `1.395 eV` |
| MB carrier-density range | `5.47e15` to `1.70e18 cm^-3` |
| Thermalized-power range | `2.21e5` to `2.98e8 W cm^-3` |
| MB validity limit | `x* ~= -2.33` for a `5%` error threshold |
| Spectra beyond that MB limit | `9 / 22` |
| Current Tsai comparison | MAE `~= 3.15 K`, bias `~= -0.68 K` |

Those numbers are not universal material parameters. They are simply the
current output of this repository for the bundled dataset and the present
settings in `hot_carrier/config.py`.

## 2. Why This Problem Is Physically Interesting

Under strong optical pumping, photoexcited carriers can remain hotter than the
lattice for long enough that a single lattice temperature is no longer an
adequate description of the electronic system. In that regime, the carrier
population is better summarized by at least two state variables:

- a carrier temperature `T`, which controls the high-energy occupation tail,
- a quasi-Fermi level splitting `Delta_mu`, which controls how far the carrier
  system sits from thermal equilibrium.

GaAs is a useful test case because it is a direct-gap semiconductor with strong
radiative recombination near the band edge. That makes photoluminescence a
practical probe of the carrier distribution. Under continuous-wave excitation,
the system reaches a steady state, so one can relate optical observables to
balance equations for generation, recombination, and cooling.

The main physical idea behind this repository is simple:

1. Fit the high-energy PL tail to extract `T` and `Delta_mu`.
2. Reconstruct the carrier state from those optical variables.
3. Use that state to estimate how absorbed power is partitioned.
4. Compare the experimental thermalized power to a microscopic cooling model.

## 3. End-to-End Workflow

`main.py` calls `hot_carrier.pipeline.main()`, which runs the following steps:

1. Validate the user configuration in `hot_carrier/config.py`.
2. Load the semicolon-separated PL matrix and sort it by increasing energy.
3. Plot the full family of raw spectra.
4. Select a high-energy fit window for each spectrum.
5. Fit the linearized generalized Planck tail.
6. Extract `T`, `Delta_mu_eff`, and `Delta_mu`.
7. Reconstruct `mu_e`, `mu_h`, and `n` with both MB and FD statistics.
8. Estimate uncertainties from line-fit covariance, fit-window sensitivity, and
   the assumed high-energy absorptivity `A0`.
9. Compute absorbed, recombination, and thermalized power channels.
10. Build the MB-validity diagnostic from the integrated generalized Planck law.
11. Run the internal Tsai-model temperature workflow if enabled.
12. Export tables and figures to `outputs/`.

That sequence is a good way to read the code as well: `pipeline.py` orchestrates
the flow, `analysis.py` contains the core optical and thermodynamic analysis,
`tsai_model.py` handles the cooling model, and `plotting.py` generates the
figures.

## 4. Data Model and Expected Inputs

### 4.1 PL spectrum file

The main input file is expected to be a semicolon-separated table with:

- the photon-energy axis in the first column, in `eV`,
- one PL intensity column per excitation condition.

The bundled file starts like this:

```text
;0;1;2;...;21
1.7712028208330721;I_0;I_1;I_2;...;I_21
1.7661566589503570;I_0;I_1;I_2;...;I_21
...
```

The code loads this file with `pandas.read_csv(..., sep=";", index_col=0)` in
`hot_carrier.analysis.load_spectra()`.

### 4.2 Excitation intensities

The excitation intensities are not read from the spectrum file. They are set
explicitly in `EXCITATION_INTENSITY_W_CM2` inside `hot_carrier/config.py`.

The number of intensity values must match the number of PL spectrum columns. The
pipeline checks this and raises an error if the lengths differ.

### 4.3 PLQY file

An optional PLQY table can be supplied through `PLQY_RESULTS_CSV`. The code
looks for columns such as:

- `phiabs (photons/s)`,
- `PLQY (%)` or `PLQY`,
- `err_PLQY (%)` or compatible uncertainty names,
- optionally an excitation-intensity column.

The mapping logic in `hot_carrier.pipeline._resolve_plqy_profiles()` is:

1. If a `phiabs` column exists and overlaps the experimental absorbed-photon
   scale, interpolate PLQY versus `log10(phiabs)`.
2. Otherwise, if an intensity column exists, interpolate PLQY versus
   `log10(intensity)`.
3. Otherwise, if the table is row-aligned with the spectra, use row order.
4. Otherwise, fall back to the constant values `PLQY_ETA` and
   `PLQY_ETA_SIGMA`.

For the bundled PLQY file and current settings, the code falls back to the
row-aligned option.

## 5. Optical Model: From PL Tail to `T` and `Delta_mu`

### 5.1 Generalized Planck law

The photoluminescence intensity is modeled with the generalized Planck law:

```text
I_PL(E) ~ [2 E^2 / (h^3 c^2)] * A(E) * {exp[(E - Delta_mu)/(k_B T)] - 1}^-1
```

where:

- `E` is photon energy,
- `A(E)` is absorptivity,
- `T` is carrier temperature,
- `Delta_mu` is quasi-Fermi level splitting.

In the high-energy tail, the Bose factor reduces to an exponential and the code
assumes that `A(E)` is locally constant over the selected fit window:

```text
I_PL(E) ~ [2 E^2 / (h^3 c^2)] * A0 * exp[-(E - Delta_mu)/(k_B T)]
```

This is the central approximation behind the optical extraction.

### 5.2 Linearization

The code fits the quantity

```text
y(E) = ln[(h^3 c^2 / (2 E^2)) * I_PL(E)] = m E + b
```

using `hot_carrier.analysis.linearized_signal()` and a straight-line regression
in `hot_carrier.analysis._compute_linear_fit_and_covariance()`.

One implementation detail matters here: the energy axis is supplied in `eV`, but
the regression itself is carried out in joules. That makes the fitted slope
consistent with the relation

```text
T = -1 / (k_B m)
```

The intercept yields an effective quasi-Fermi level splitting:

```text
Delta_mu_eff = (b k_B T) / e
```

and the code then corrects for the assumed high-energy absorptivity `A0`:

```text
Delta_mu = Delta_mu_eff - (k_B T / e) ln(A0)
```

This distinction between `Delta_mu_eff` and `Delta_mu` is deliberate. The first
contains the absorptivity contribution implicitly. The second is the reported
QFLS after the nominal `A0` correction.

### 5.3 What the slope and intercept really mean

The temperature comes from the slope, so it is mainly controlled by the shape of
the high-energy PL tail. `Delta_mu`, by contrast, comes from the intercept. That
means it is much more sensitive to:

- absolute PL intensity calibration,
- detector-response corrections,
- the assumed absorptivity `A0`,
- any hidden multiplicative scale factor in the measured intensity.

That sensitivity propagates directly into `mu_e`, `mu_h`, and carrier density.
For this reason, the temperature extraction is usually the most robust part of
the optical fit, while density-related quantities require more caution.

## 6. Fit-Window Selection and Uncertainties

### 6.1 Automatic fit-window selection

If `AUTO_SELECT_FIT_WINDOW = True`, the code searches within
`WINDOW_SEARCH_MIN_EV` to `WINDOW_SEARCH_MAX_EV`. It can optionally start the
search above the PL peak by adding `WINDOW_PEAK_OFFSET_EV`.

Every contiguous candidate window is tested and kept only if it satisfies the
configured conditions:

- enough data points,
- positive intensities,
- negative fitted slope,
- minimum `R^2`,
- optional physical temperature bounds.

The surviving windows are scored with the corrected Akaike information criterion
(AICc):

```text
AIC  = N ln(RSS/N) + 2k
AICc = AIC + [2k(k + 1)] / (N - k - 1)
```

with `k = 2` for slope and intercept. The selected fit window is the one with
the smallest AICc.

This logic is implemented in:

- `hot_carrier.analysis._build_scan_candidate_mask()`,
- `hot_carrier.analysis._enumerate_window_fit_samples()`,
- `hot_carrier.analysis.auto_select_fit_window()`,
- `hot_carrier.analysis.fit_single_spectrum()`.

### 6.2 Uncertainty model

The code reports three uncertainty components for each fitted parameter:

```text
sigma_total^2 = sigma_chi2^2 + sigma_range^2 + sigma_A0^2
```

The three pieces are:

- `chi2`: covariance of the linear regression coefficients,
- `range`: sensitivity to plausible alternative fit windows,
- `A0`: uncertainty induced by the allowed high-energy absorptivity range.

The fit-window term is not a Bayesian posterior. It is an objective
window-sensitivity metric built from the AICc-weighted spread of acceptable
windows around the chosen one.

Importantly, the reported `range` uncertainty is computed from the full
AICc-weighted ensemble of acceptable windows. The setting
`FIT_RANGE_SCAN_PLOT_WEIGHT_COVERAGE` only controls how much cumulative AICc
weight is shown in the plotted fit-window envelope, with `0.95` used as a
visual default. It does not truncate the uncertainty calculation itself.

The uncertainties are exported as separate columns in `fit_results.csv`, for
example:

- `temperature_err_chi2_k`,
- `temperature_err_range_k`,
- `temperature_err_a0_k`,
- `temperature_err_total_k`.

The same naming pattern is used for `qfls`, the MB quantities `mu_e`, `mu_h`,
and carrier density, and the FD quantities `mu_e_fd`, `mu_h_fd`, and
`carrier_density_fd`.

## 7. Carrier-State Reconstruction

### 7.1 Band model and reference energies

The code assumes parabolic conduction and valence bands and computes the
effective density of states as:

```text
N_c(T) = 2 * [(m_e* k_B T)/(2 pi hbar^2)]^(3/2)
N_v(T) = 2 * [(m_h* k_B T)/(2 pi hbar^2)]^(3/2)
```

The reported `mu_e` and `mu_h` are referenced to mid-gap, so the conduction and
valence band edges sit at `+E_g/2` and `-E_g/2` in that convention.

### 7.2 Maxwell-Boltzmann reconstruction

Using charge neutrality `n = p` and `Delta_mu = mu_e + mu_h`, the code computes:

```text
mu_e = 0.5 * [Delta_mu - (k_B T / e) ln(N_c/N_v)]
mu_h = 0.5 * [Delta_mu + (k_B T / e) ln(N_c/N_v)]
n    = N_c * exp[((mu_e - E_g/2) e)/(k_B T)]
```

This is implemented in `hot_carrier.analysis.compute_mu_and_density_mb()`.

### 7.3 Fermi-Dirac reconstruction

The repository also solves the charge-neutral FD problem using the complete
Fermi-Dirac integral of order one-half and a bisection search. This is
implemented in:

- `hot_carrier.analysis._fermi_dirac_half()`,
- `hot_carrier.analysis.compute_mu_and_density_fd()`.

The optical fit itself does not change when FD is used. The fitted `T` and
`Delta_mu` come from the PL tail either way. FD only changes the back-calculated
carrier-state quantities `mu_e`, `mu_h`, and `n`.

The repository now also propagates the same three optical uncertainty
contributions into the FD state variables:

- line-fit covariance (`chi2`),
- fit-window sensitivity (`range`),
- absorptivity uncertainty (`A0`).

### 7.4 Why both MB and FD are reported

MB is convenient, fast, and analytically transparent. FD is more reliable once
the state approaches degeneracy. In this repository:

- both MB and FD carrier reconstructions are exported with propagated
  uncertainties derived from the same fitted `T` and `Delta_mu`,
- FD values are now the default quantities used in the per-carrier
  power-balance terms and in the `Delta_mu`-based Tsai inversion,
- MB companion values are still exported so the statistical closure can be
  compared explicitly in the thermalized-power and Tsai diagnostics,
- FD values show how much the reconstructed carrier state shifts when
  degeneracy matters.

That comparison becomes important at the high-intensity end of the dataset.

## 8. MB Validity Diagnostic for the Photon Occupation

The repository includes a separate MB-validity test based on the integrated
generalized Planck law, not just on the carrier-density formulas.

The code evaluates:

```text
Phi = integral_{E_g}^infinity I_PC(E) dE
```

for a step absorber:

```text
A(E) = A0 * Theta(E - E_g)
```

and compares:

- the exact Bose-Einstein photon occupation,
- the first Boltzmann term used in the MB approximation.

The diagnostic is expressed in terms of the reduced variable

```text
x = (Delta_mu - E_g) / (k_B T)
```

The code then defines a threshold `x*` as the first point where the relative
error

```text
epsilon_MB = Phi_BE / Phi_MB - 1
```

exceeds `MB_VALIDITY_REL_ERROR_LIMIT`, which is `5%` by default.

Implementation:

- `hot_carrier.analysis.integrated_ipc_step_absorber_be()`,
- `hot_carrier.analysis.integrated_ipc_step_absorber_mb()`,
- `hot_carrier.analysis.build_mb_validity_scan()`.

For the current bundled dataset and configuration:

- the conservative boundary is `x* ~= -2.33`,
- the experimental points span roughly `x = -11.76` to `x = -0.94`,
- `9` of the `22` spectra lie beyond that `5%` MB limit.

That is an important result. It means the MB back-extraction is still useful as
an internally consistent baseline, but the high-intensity end of the dataset is
already entering a regime where FD-based carrier interpretation is the safer
choice.

## 9. Power-Balance Model

The repository uses a compact steady-state power-balance model. The starting
relations are:

```text
phi_abs = phi_rad + phi_nonrad
P_abs   = P_th + P_rad + P_nonrad
```

with:

```text
P_abs   = A_laser * P_exc
phi_abs = P_abs / E_laser
phi_rad = eta * phi_abs
phi_nonrad = (1 - eta) * phi_abs
```

The recombination-channel energies are modeled as:

```text
E_nonrad = E_g + 3 k_B T
E_rad    = E_g + k_B T
```

so that:

```text
P_rec = phi_nonrad * E_nonrad + phi_rad * E_rad
P_th  = P_abs - P_rec
```

Area-normalized powers are converted to volumetric powers with the active-layer
thickness `d`:

```text
P_vol = P_area / d_cm
```

This model is implemented in `hot_carrier.analysis.compute_power_balance_table()`.
The function also propagates first-order uncertainties from:

- absorptivity at the laser wavelength,
- PLQY,
- extracted temperature.

The volumetric power channels themselves do not depend on MB versus FD carrier
statistics. The statistics choice only enters the derived per-carrier cooling
quantities through the carrier density. The code now exports both:

- `thermalized_power_per_carrier_mb_ev_s`,
- `thermalized_power_per_carrier_fd_ev_s`,

and uses FD by default for the generic
`thermalized_power_per_carrier_ev_s` column through
`POWER_BALANCE_CARRIER_STATISTICS = "fd"`.

The result is a useful energy-partition diagnostic, but it is intentionally
compact. It is not a full transport model, and it does not include spatial
gradients, diffusion, or a microscopic treatment of all recombination channels.

## 10. Tsai Eq. 41 Plus Eq. 48 Cooling Workflow

### 10.1 What this section is really about

The repository does **not** implement Tsai's full photovoltaic-device model in
the same self-consistent way as the paper. What it does implement is the
electron intraband cooling kernel from the paper and then compare that kernel to
the experimentally inferred thermalized-power channel.

That distinction matters.

In the paper, the starting point is the carrier energy-density balance
(Tsai Eq. 3):

```text
du/dt = (1/L) * (p_gen - p_rec - J·V) + (du/dt)_intra
```

where:

- `u` is carrier energy density,
- `p_gen` is the incoming optical power flux,
- `p_rec` is the interband recombination power flux,
- `J·V` is the extracted electrical power flux,
- `L` is the active-region thickness,
- `(du/dt)_intra` is the intraband carrier cooling term, dominated here by
  electron interaction with LO phonons.

At steady state, Eq. 3 becomes

```text
- (du/dt)_intra = (p_gen - p_rec - J·V) / L
```

Tsai uses that relation inside a device model. This repository uses it in a
more restricted way:

1. The optical analysis first infers an experimental thermalized power density
   `P_th_exp` from the PL data.
2. The Tsai workflow computes a modeled intraband cooling power density
   `P_th_model = - (du/dt)_intra`.
3. The comparison is then made through the identification

```text
P_th_exp  ~=  - (du/dt)_intra
```

This is scientifically useful, but it is not identical to solving Tsai's full
Eq. 3 together with a full transport and `J-V` model.

### 10.2 Physical picture in plain language

The physical idea is simple:

- hot carriers have excess kinetic energy above the lattice temperature,
- in GaAs, electrons lose that excess energy mainly by emitting longitudinal
  optical (LO) phonons,
- those LO phonons do not disappear instantly,
- if the LO phonons live long enough, they build up a nonequilibrium
  population and can be reabsorbed,
- that reabsorption slows carrier cooling.

So the cooling rate is controlled by a competition:

- faster electron -> LO phonon emission increases cooling,
- stronger screening weakens the electron-phonon interaction and decreases
  cooling,
- longer LO-phonon lifetime makes hot-phonon reabsorption stronger and also
  decreases cooling.

This is why Tsai's model is more realistic than a single fixed energy-relaxation
time. The cooling rate is not a constant. It depends on carrier temperature,
carrier density, chemical potential, screening, and LO-phonon lifetime.

### 10.3 Which equations from Tsai are used

The implementation follows the paper's hierarchy:

```text
Eq. 3  ->  Eq. 48  ->  Eq. 41  ->  Eq. 35  ->  Eq. 34
```

with the steady-state hot-phonon steps of Eqs. 39, 40, 46, and 47 used in
between.

#### Eq. 34: bare polar-LO interaction

For the channel actually used in this repository, Tsai keeps only the polar LO
phonon matrix element from Eq. 34:

```text
|M_q|^2 = [e^2 ħω_LO / (2 ε0 q^2 V)] * (1/ε_inf - 1/ε_static)
```

This is the Fröhlich interaction for polar optical phonons.

Interpretation:

- the coupling is strongest at small `q` because of the `1/q^2` factor,
- it is stronger in more polar materials through
  `(1/ε_inf - 1/ε_static)`,
- only the LO channel is kept because Tsai argues, and the code follows, that
  it dominates energy relaxation in GaAs.

In code, this is the `base` term in
`hot_carrier.tsai_model._screened_matrix_element_sq_times_volume_j2()`.

#### Eq. 35 plus Eq. 37-38: screening

Tsai then screens the matrix element through Eq. 35:

```text
|M_q^screen|^2 = |M_q|^2 * |ε0 / ε_screen(q, ω)|^2
```

The repository does **not** use the full dynamic Lindhard dielectric function of
Eq. 36. It uses Tsai's static Thomas-Fermi limit from Eqs. 37-38:

```text
ε_screen(q, 0) = ε0 * (1 + q_s^2 / q^2)
q_s^2 = [e^2 / (ε0 ε_static)] * (∂n/∂μ)
```

Therefore the code evaluates

```text
|M_q^screen|^2 = |M_q|^2 / (1 + q_s^2 / q^2)^2
```

The screening wave number `q_s` is computed in
`hot_carrier.tsai_model._screening_wave_number_m1()`.

Two distinct approximations appear here:

- `TSAI_SCREENING_MODEL = "mb"`:

```text
∂n/∂μ = n / (k_B T)
```

- `TSAI_SCREENING_MODEL = "fd_fdiff"`:
  `∂n/∂μ` is evaluated numerically by finite differences from the FD carrier
  density.

This is separate from the choice of how `mu_e` is reconstructed from
`Delta_mu`. Those are two different model layers.

#### Eq. 41: mode-resolved LO-phonon emission time

With the screened interaction, Tsai derives the explicit emission time for each
phonon wave number `q`:

```text
1/τ_c-LO^q =
    [m_c^2 k_B T_c |M_q^screen|^2 V_c / (π ħ^5 q)]
    * ln[(1 + exp(η_c - ε_min + ε_LO)) / (1 + exp(η_c - ε_min))]
```

with

```text
η_c   = μ_c / (k_B T_c)
ε_LO  = ħω_LO / (k_B T_c)
ε_min = ħ^2 k_min^2 / (2 m_c k_B T_c)
k_min = q/2 + m_c ω_LO / (ħ q)
```

Interpretation:

- `τ_c-LO^q` is the characteristic time for electrons to emit LO phonons of a
  given wave number `q`,
- the logarithm is the Fermi-statistics phase-space factor,
- `ε_min` enforces the kinematic threshold for a carrier to emit an LO phonon
  while conserving energy and momentum,
- stronger screening increases `τ_c-LO^q`, meaning slower cooling.

This is implemented in `hot_carrier.tsai_model._tau_c_lo_q_seconds()`.

One important project-specific detail is that the repository stores `mu_e`
relative to mid-gap, while Eq. 41 uses the electron chemical potential relative
to the conduction-band edge. The code therefore converts

```text
μ_c = μ_e - E_g/2
```

before evaluating Eq. 41.

#### Eq. 48: total intraband cooling rate with hot-phonon bottleneck

Tsai then combines carrier emission with LO-phonon decay and obtains the
steady-state intraband cooling rate:

```text
(du/dt)_intra =
    - (1 / 2π^2) ∫ dq q^2 ħω_LO [N_q(T_c) - N_q(T_L)] / [τ_c-LO^q + τ_LO]
```

where

```text
N_q(T) = 1 / [exp(ħω_LO / k_B T) - 1]
```

is the Bose-Einstein occupation from Eq. 40.

This expression is the compact result of the hot-phonon part of the paper:

- Eq. 39 introduces carrier-driven LO-phonon emission,
- Eq. 44 introduces LO-phonon decay,
- Eq. 46 gives the steady-state nonequilibrium phonon population,
- Eq. 47 gives the net LO emission rate,
- Eq. 48 inserts that result into the carrier energy-loss integral.

Interpretation:

- if `T_c > T_L`, then `N_q(T_c) > N_q(T_L)`, so carriers tend to emit net LO
  phonons and lose energy,
- if `τ_LO` is long, the denominator becomes larger and cooling slows down,
- if `τ_c-LO^q` is large because screening is strong, cooling also slows down.

In code, Eq. 48 is implemented in
`hot_carrier.tsai_model.compute_du_dt_intra_electron_w_cm3()`.

The code also reports an effective energy-relaxation time associated with the
right-hand side of Eq. 48:

```text
τ_E = [u(T_c) - u(T_L)] / [-(du/dt)_intra]
```

In implementation terms, `tau_c_lo_energy_s` is obtained as the ratio of the
Eq. 48 numerator-like integral to `-(du/dt)_intra` over the same numerical `q`
grid, so it should be read as the code's effective Tsai-style relaxation time.

### 10.4 How the paper equations are translated into code

The implementation is deliberately narrower than the full paper:

- only **electrons** are cooled explicitly,
- only **polar LO phonons** are retained,
- screening is **static Thomas-Fermi**, not full dynamic Lindhard,
- the `q` integral from `0` to `∞` is approximated on a finite geometric grid
  from `TSAI_Q_MIN_CM1` to `TSAI_Q_MAX_CM1`,
- the LO-phonon lifetime is supplied as an effective input
  `TSAI_LO_PHONON_LIFETIME_PS` rather than computed microscopically from
  Eq. 45,
- the code evaluates the steady-state closed form of Eq. 48 directly instead of
  evolving the phonon population dynamically in time.

The main implementation path is:

- `_screened_matrix_element_sq_times_volume_j2()`: Eq. 34 plus static-screening
  reduction,
- `_screening_wave_number_m1()`: Eq. 38 under MB or FD finite-difference
  closure,
- `_tau_c_lo_q_seconds()`: Eq. 41,
- `compute_du_dt_intra_electron_w_cm3()`: Eq. 48,
- `_compute_forward_grid()`: evaluates the model on a `(state, T)` grid,
- `_build_inverse_grid()` and `_build_temperature_predictor()`: numerically
  invert the forward map,
- `run_tsai_temperature_workflow()`: compare the inverse model to the
  experiment.

The sign convention is:

```text
(du/dt)_intra < 0  for cooling
P_th_model        = max( - (du/dt)_intra , 0 )
```

so all plotted thermalized powers are non-negative.

### 10.5 Why the default primary axis is `Delta_mu`

By default, `TSAI_PRIMARY_INPUT = "delta_mu"`. This choice is a practical
anti-circularity measure.

If `mu_e` were used directly as the primary input, one would feed the Tsai model
a quantity that is already reconstructed from the optical fit and therefore
already depends on the measured carrier temperature.

Instead, the default workflow uses the experimental `Delta_mu` and reconstructs
`mu_e(Delta_mu, T)` internally for each trial temperature on the Tsai grid:

```text
Delta_mu, T  ->  mu_e, n  ->  q_s  ->  τ_c-LO^q  ->  (du/dt)_intra
```

That reconstruction uses the same equal-injection carrier-state model as the
rest of the repository:

- MB closure from `analysis.compute_mu_and_density_mb()`, or
- FD closure from `analysis.compute_mu_and_density_fd()`.

The current default is
`TSAI_DELTA_MU_CARRIER_STATISTICS = "fd"`, but MB comparison columns are also
exported.

This reduces circularity, but it does **not** remove all model dependence.
`Delta_mu` itself still comes from the optical fit, and the `Delta_mu -> mu_e`
mapping still depends on the assumed band structure and carrier statistics.

### 10.6 How the comparison to experiment is actually done

The experimental side of the comparison comes from the earlier optical and
power-balance stages of the repository. For each spectrum, the code already has:

- `temperature_k_exp` from the PL-tail fit,
- `delta_mu_ev` from the same optical reconstruction,
- `p_th_exp_w_cm3 = thermalized_power_w_cm3` from the power-balance model.

The crucial identification is:

```text
p_th_exp_w_cm3  ~=  p_th_model_w_cm3  =  - (du/dt)_intra
```

The workflow is then:

1. Build a grid in either `(Delta_mu, T)` or `(mu_e, T)`.
2. Compute `p_th_model_w_cm3` on that grid from Eq. 48.
3. For each fixed state value, invert the monotonic relation

```text
(state, T) -> p_th_model
```

   into

```text
(state, p_th_model) -> T
```

   using interpolation in `log10(p_th)`.
4. Evaluate that inverse map at the experimental pairs

```text
(delta_mu_exp, p_th_exp)
```

   or

```text
(mu_e_exp, p_th_exp)
```

   to obtain `temperature_sim_k`.
5. Compare `temperature_sim_k` with the measured `temperature_k_exp`.

This is why the Tsai workflow is an **inverse consistency test**:

- if the paper-based cooling model predicts roughly the same temperature as the
  optical fit, then the inferred thermalized power is consistent with that
  cooling law,
- if it does not, then some combination of the cooling model, screening choice,
  phonon lifetime, or optical power-balance assumptions is inconsistent.

### 10.7 What the exported Tsai files mean

The main products are:

- `outputs/tsai_forward_stateT_to_pth.csv`:
  the forward map `(state, T) -> p_th_model`,
- `outputs/tsai_inverse_pth_state_to_temperature.csv`:
  the numerically inverted map `(state, p_th) -> T`,
- `outputs/tsai_du_dt_samples_at_experimental_state.csv`:
  Eq. 48 evaluated directly at the experimental state points,
- `outputs/tsai_temperature_comparison.csv`:
  simulated temperatures, residuals, and reconstructed state variables,
- `outputs/tsai_temperature_rise_vs_pth_density.png`:
  the main visual comparison.

When `TSAI_PRIMARY_INPUT = "delta_mu"`, the comparison table includes both MB
and FD reconstructions of:

- experimental `mu_e` and carrier density,
- simulated `mu_e` and carrier density,
- Tsai-predicted temperatures and residuals.

With the bundled dataset and current defaults:

- `TSAI_LO_PHONON_LIFETIME_PS = 16.0`,
- `TSAI_Q_MIN_CM1 = 3e4`,
- `TSAI_Q_MAX_CM1 = 1e8`,
- `TSAI_SCREENING_MODEL = "mb"`,
- `TSAI_DELTA_MU_CARRIER_STATISTICS = "fd"`,

the present comparison gives approximately:

- FD default closure: MAE `~= 3.31 K`, bias `~= -1.76 K`,
- MB comparison closure: MAE `~= 3.15 K`, bias `~= -0.68 K`.

### 10.8 What this means scientifically, and what it does not mean

If the Tsai-predicted and experimentally inferred temperatures agree reasonably
well, the correct conclusion is:

- the experimentally inferred thermalized-power density is broadly consistent
  with an electron-LO-phonon cooling model of the Tsai type for the chosen
  parameters and closures.

The correct conclusion is **not**:

- that the Tsai paper has been reproduced in full,
- that the LO-phonon lifetime or screening model has been uniquely identified,
- that the optical `P_th` extraction is model-free,
- that all carrier heating in the sample is necessarily electron-only LO-phonon
  cooling.

The most important limitations are:

- Eq. 3 is **not** solved here as a full device balance including a computed
  `J·V` term,
- the experimental `P_th` comes from the repository's optical power-balance
  model, not from a direct calorimetric measurement,
- holes are not cooled explicitly in the Tsai stage,
- dynamic screening and coupled plasmon-phonon modes are not included,
- `tau_LO` is treated as an effective parameter,
- the inverse comparison inherits uncertainty from absorptivity, PLQY,
  thickness, the optical fit, and the MB/FD carrier-state closure.

That is why the Tsai workflow should be interpreted as a rigorous but still
model-dependent consistency check between:

- the optical state extracted from PL, and
- a literature-based microscopic cooling law for electron -> LO-phonon energy
  relaxation.

## 11. Software Architecture

The repository is small enough that each file has a clear role:

- `main.py`: minimal entry point that calls `hot_carrier.pipeline.main()`.
- `hot_carrier/config.py`: user-editable settings and physical constants.
- `hot_carrier/pipeline.py`: top-level orchestration, file I/O, and output
  export.
- `hot_carrier/analysis.py`: optical fitting, MB and FD reconstruction,
  uncertainties, MB-validity scan, and power balance.
- `hot_carrier/tsai_model.py`: Tsai cooling model plus forward and inverse
  grids.
- `hot_carrier/plotting.py`: all figures and diagnostics.
- `hot_carrier/models.py`: data classes used to assemble fit results cleanly.

There is no command-line interface at the moment. The expected way to use the
project is to edit `hot_carrier/config.py` and rerun `main.py`.

## 12. Important Configuration Groups

Most users only need to touch a small number of configuration blocks:

- Data source: `FILENAME`, `EXCITATION_INTENSITY_W_CM2`
- Optical fit: `AUTO_SELECT_FIT_WINDOW`, `FIT_ENERGY_*`, `WINDOW_*`
- High-energy absorptivity: `A0_HIGH_ENERGY_MIN`, `A0_HIGH_ENERGY_MAX`,
  `A0_UNCERTAINTY_MODEL`, `ASSUMED_A0`
- Material parameters: `EG_EV`, `M_E_EFF`, `M_H_EFF`
- Power balance: `LASER_WAVELENGTH_NM`, `ABSORPTIVITY_AT_LASER`, `PLQY_ETA`,
  `PLQY_RESULTS_CSV`, `ACTIVE_LAYER_THICKNESS_NM`
- MB diagnostic: `MB_VALIDITY_*`
- Tsai workflow: `TSAI_ENABLE_SIMULATION`, `TSAI_PRIMARY_INPUT`,
  `TSAI_DELTA_MU_CARRIER_STATISTICS`, `TSAI_LO_*`, `TSAI_Q_*`,
  `TSAI_SCREENING_*`, `TSAI_*_GRID_*`

Two settings are especially important from a physics point of view:

- `ASSUMED_A0`, because it directly shifts the reported `Delta_mu`,
- `PLQY_RESULTS_CSV` or `PLQY_ETA`, because they strongly affect the
  recombination and thermalized-power partition.

## 13. Environment and How to Run

The current checked virtual environment uses Python `3.14.3`. The code imports:

- `numpy`,
- `pandas`,
- `matplotlib`,
- `scipy`.

To run from the bundled environment:

```powershell
.\.venv\Scripts\python.exe main.py
```

To recreate a clean Windows environment from scratch:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pandas matplotlib scipy
python main.py
```

After a successful run, the pipeline rewrites the exported CSV files and figures
in `outputs/`.

## 14. Outputs and How to Interpret Them

### 14.1 Core outputs

- `outputs/fit_results.csv`: main per-spectrum results table.
- `outputs/all_spectra_logscale.png`: raw PL spectra, colored by excitation
  intensity.
- `outputs/fits/fit_spectrum_XX.png`: per-spectrum raw fit, selected window,
  and linearized tail.
- `outputs/parameters_vs_intensity.png`: `T`, `Delta_mu`, `mu_e`, `mu_h`, and
  `n` versus excitation intensity.
- `outputs/thermalized_power_diagnostics.png`: power diagnostics in state space
  and per-carrier form, including MB-vs-FD carrier-density comparisons.
- `outputs/recombination_channel_contributions.png`: direct comparison of
  `P_rad` and `P_nonrad`, plus the `eta`-driven change in `P_rec` and `P_th`
  relative to the `eta = 0` limit.
- `outputs/mb_validity_limit.png`: exact BE versus MB integrated-GPL
  comparison.
- `outputs/mb_validity_scan.csv`: full MB-validity scan versus reduced
  `Delta_mu`.
- `outputs/mb_validity_limits.csv`: detected MB threshold `x*` for each
  reference temperature.

### 14.2 Tsai outputs

- `outputs/tsai_forward_stateT_to_pth.csv`: forward map from state and
  temperature to modeled `P_th`.
- `outputs/tsai_inverse_pth_state_to_temperature.csv`: inverse temperature map.
- `outputs/tsai_du_dt_samples_at_experimental_state.csv`: cooling-rate samples
  evaluated at experimental states.
- `outputs/tsai_temperature_comparison.csv`: experimental and simulated
  temperatures, errors, and reconstructed MB/FD state variables.
- `outputs/tsai_temperature_rise_vs_pth_density.png`: main figure of merit,
  `T - T_L` versus `P_th`, with MB and FD Tsai closures compared when
  `TSAI_PRIMARY_INPUT = "delta_mu"`.

### 14.3 The most important columns in `fit_results.csv`

The results table is wide, but a small set of columns does most of the work:

- optical fit quality:
  `fit_min_ev`, `fit_max_ev`, `window_mode`, `n_points_fit`, `r2`,
  `fit_range_samples`,
- extracted state:
  `temperature_k`, `qfls_effective_ev`, `qfls_ev`,
  `mu_e_ev`, `mu_h_ev`, `carrier_density_cm3`,
- FD comparison:
  `mu_e_fd_ev`, `mu_h_fd_ev`, `carrier_density_fd_cm3`,
- uncertainty columns:
  all `*_err_chi2_*`, `*_err_range_*`, `*_err_a0_*`, `*_err_total_*`,
  including `mu_e_fd_err_total_ev`, `mu_h_fd_err_total_ev`,
  and `carrier_density_fd_err_total_cm3`,
- power-balance outputs:
  `absorbed_power_*`, `recombination_power_*`, `thermalized_power_*`,
  `thermalized_power_per_carrier_mb_*`, `thermalized_power_per_carrier_fd_*`,
  `thermalized_fraction`, `recombination_fraction`,
  `radiative_fraction`, `nonradiative_fraction`,
- closure checks:
  `power_balance_closure_*`, `carrier_balance_closure_cm2_s`.

Small closure values indicate that the algebraic bookkeeping is internally
consistent.

## 15. What the Current Results Suggest

For the bundled dataset and present settings, the outputs tell a coherent story:

- the extracted carrier temperature rises only moderately above room
  temperature, from about `299 K` to `378 K`,
- `Delta_mu` increases toward the band gap, reaching about `1.395 eV` while
  `E_g` is set to `1.424 eV`,
- carrier density rises strongly with excitation, reaching the `10^18 cm^-3`
  range,
- the MB-validity scan shows that the highest-intensity spectra are already on
  the non-MB side of the repository's own `5%` integrated-photon criterion,
- the current Tsai implementation reproduces the measured temperature rise
  reasonably well for this dataset once the current GaAs-specific parameters are
  used.

The important scientific caution is that the MB and Tsai conclusions are not
fully independent. In the current workflow, the Tsai inversion reconstructs
`mu_e` from the same optical `T` and `Delta_mu` pair used for the carrier-state
post-processing. The default closure is now FD, but the inferred agreement with
Tsai still depends on that thermodynamic reconstruction and on the separate
screening choice.

## 16. Current Limitations and the Most Useful Next Steps

### 16.1 Present limitations

- The optical extraction assumes a locally constant absorptivity `A(E) ~= A0`
  over the fit window.
- The MB-validity diagnostic uses a step absorber, so it does not capture the
  real spectral structure of GaAs absorptivity.
- `Delta_mu` depends on the PL intercept and is therefore sensitive to absolute
  intensity calibration and to the assumed `A0`.
- The Tsai workflow is electron-only and does not represent a full coupled
  electron-hole transport or recombination model.
- Even with FD as the default `Delta_mu -> mu_e` closure, the Tsai workflow
  still relies on a separate screening approximation, which remains MB by
  default unless `TSAI_SCREENING_MODEL` is changed to `fd_fdiff`.
- The repository has no automated test suite, no saved run manifest, and no
  package metadata or dependency lockfile.

### 16.2 Highest-value improvements

If this project is going to evolve into a more reliable research tool, the most
useful next steps are:

1. Add a reproducibility manifest to every run.

   Save the full configuration, Python version, package versions, and git commit
   hash alongside the outputs. This is the lowest-effort improvement with the
   highest immediate value.

2. Add regression tests on the bundled dataset.

   A small test suite should verify that `fit_results.csv`,
   `mb_validity_limits.csv`, and the main Tsai metrics stay numerically stable
   when the code changes.

3. Propagate uncertainties through the Tsai stage.

   Right now the uncertainty story covers the optical fit, MB and FD state
   reconstruction, and the compact power balance. The next missing piece is
   uncertainty on the simulated Tsai temperatures.

4. Replace constant `A0` with a spectral `A(E)` treatment.

   That would improve both the optical extraction and the MB-validity analysis.
   It is one of the most important physics upgrades available.

5. Move from tuned parameters to joint inference.

   Parameters such as `A0`, PLQY, `tau_LO`, and screening choice are currently
   set externally or tuned manually. A joint inference workflow would make the
   results much more defensible.

6. Extend the cooling model beyond the current electron-only implementation.

   A fuller treatment should eventually include hole cooling, coupled carrier
   dynamics, and possibly additional energy-loss channels.

7. Replace file-edited configuration with a manifest or CLI.

   Editing `hot_carrier/config.py` is workable for a single user, but it is not
   ideal for batch studies or collaborative use.

## 17. Literature Included in the Repository

The `literature/` folder contains the local reference material used to frame the
project, including:

- `Tsai_2018_Hot_Carrier_Model.pdf`,
- `Vezin_2024_CWPL.pdf`,
- `Vezin_2025_Hot_Electrons.pdf`,
- `Vezin_2025_Distinguishing.pdf`,
- `Giteau_2020_Hot_Carriers.pdf`,
- related supporting notes and papers.

Those files are useful for checking the model assumptions and for extending the
code, especially the CWPL interpretation and the Tsai cooling section.
