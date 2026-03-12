# Hot-Carrier Analysis from Continuous-Wave Photoluminescence in GaAs

This repository analyzes calibrated continuous-wave photoluminescence (CWPL)
spectra from bulk GaAs and extracts a steady-state hot-carrier state from them.
The code does not treat the spectra as generic lineshapes. It assumes that, in
the high-energy tail, the emission follows the generalized Planck law and that
the carrier population can be summarized by a common carrier temperature `T` and
a quasi-Fermi level splitting `Delta_mu`. From those two optical quantities it
reconstructs electron and hole chemical potentials, carrier density, absorbed
and thermalized power, and finally compares the inferred thermalized power with
an electron-LO-phonon cooling model based on Tsai 2018.

This README is written as a methods report. For each step it states what the
code does, why that choice is physically reasonable for the bundled dataset, and
what the approximation leaves out.

## 1. Scope and Current Numerical Regime

The repository currently contains one bundled GaAs CWPL dataset,
`GaAs bulk_PL_avg_circle_4pixs.txt`, one PLQY table,
`GaAs bulk_PLQY_results.csv`, the analysis code under `hot_carrier/`, and the
reference material under `literature/`.

With the current configuration in `hot_carrier/config.py` and the bundled
outputs in `outputs/`, the analysis operates in the following regime:

| Quantity | Current value |
| --- | --- |
| Number of spectra | `22` |
| Energy samples per spectrum | `141` |
| Measured photon-energy range | `1.2651` to `1.7712 eV` |
| Excitation-intensity range | `9.46e1` to `1.30e5 W cm^-2` |
| PL peak-energy range | `1.4251` to `1.4350 eV` |
| Selected fit-window lower bound | `1.4586` to `1.5537 eV` |
| Selected fit-window upper bound | `1.7031` to `1.7662 eV` |
| Selected points per fit | `49` to `71` |
| Fit quality `R^2` | `0.99958` to `0.99995` |
| Extracted temperature range | `298.9` to `378.1 K` |
| Extracted `Delta_mu` range | `1.1212` to `1.3947 eV` |
| MB carrier-density range | `5.47e15` to `1.70e18 cm^-3` |
| FD carrier-density range | `5.45e15` to `1.20e18 cm^-3` |
| PLQY range used in power balance | `0.23` to `5.80 %` |
| Thermalized-power range | `2.21e5` to `2.98e8 W cm^-3` |
| Thermalized fraction of absorbed power | `34.8` to `35.6 %` |
| Conservative MB-validity limit | `x* ~= -2.33` at `5 %` error |
| Experimental reduced-QFLS range | `-11.76` to `-0.94` |
| Spectra beyond that MB limit | `9 / 22` |
| Tsai inverse comparison, FD default | MAE `~= 3.31 K`, bias `~= -1.76 K` |
| Tsai inverse comparison, MB comparison column | MAE `~= 3.15 K`, bias `~= -0.68 K` |

These numbers are not universal GaAs constants. They are the present outputs of
this specific code, with this specific dataset, under the current settings.

## 2. Scientific Question and Why PL Can Answer It

Under strong continuous-wave excitation, the electronic subsystem can remain
hotter than the lattice even though the sample has reached a macroscopic steady
state. In that regime, the lattice temperature alone is not enough to describe
the emitted light. A more useful reduced description is:

- a carrier temperature `T`, which controls the decay of the high-energy PL
  tail,
- a quasi-Fermi level splitting `Delta_mu`, which measures how far the carrier
  population is driven from equilibrium.

Bulk GaAs is a good system for this analysis for three reasons:

1. It is a direct-gap semiconductor, so radiative recombination near the band
   edge is strong enough to measure accurately.
2. The bundled spectra are already calibrated in photon units, so the PL
   intercept carries physical meaning rather than just arbitrary detector counts.
3. The steady-state CW condition makes it reasonable to write balance equations
   between absorbed power, recombination power, and thermalized power.

The code therefore follows a simple research logic:

1. Extract `T` and `Delta_mu` from the high-energy PL tail.
2. Convert `(T, Delta_mu)` into `(mu_e, mu_h, n)`.
3. Estimate how the absorbed laser power is partitioned between recombination
   and thermalization.
4. Ask whether that inferred thermalized power is consistent with a microscopic
   electron cooling model.

## 3. End-to-End Code Path

`main.py` is intentionally minimal. It just calls
`hot_carrier.pipeline.main()`. The pipeline then runs the following sequence:

1. Validate the user configuration in `hot_carrier/config.py`.
2. Load the semicolon-separated PL matrix with
   `hot_carrier.analysis.load_spectra()`.
3. Sort the energy axis in ascending order.
4. Plot the raw spectra.
5. Choose a high-energy fit window for each spectrum.
6. Fit the linearized generalized Planck tail.
7. Extract `T`, `Delta_mu_eff`, and `Delta_mu`.
8. Reconstruct `mu_e`, `mu_h`, and `n` with both MB and FD carrier statistics.
9. Build an uncertainty budget from line-fit covariance, fit-window
   sensitivity, and `A0`.
10. Compute absorbed, radiative, nonradiative, recombination, and thermalized
    power channels.
11. Diagnose when the MB approximation for the photon occupation has broken
    down.
12. Run the Tsai electron cooling workflow if enabled.
13. Export all figures and CSV tables.

The scientific work is concentrated in two modules:

- `hot_carrier/analysis.py`: optical extraction, carrier reconstruction,
  uncertainty propagation, MB-validity scan, and power balance.
- `hot_carrier/tsai_model.py`: Tsai Eq. 41 plus Eq. 48 cooling kernel, forward
  map, inverse map, and temperature comparison.

## 4. Inputs, Units, and Implicit Experimental Assumptions

### 4.1 PL spectra

The main data file is read as:

```text
pandas.read_csv(..., sep=";", index_col=0)
```

The first column is the photon-energy axis in `eV`. Each additional column is a
single spectrum measured at one excitation intensity.

The code assumes:

- the intensities are strictly proportional to emitted photon flux,
- the detector response has already been corrected,
- the energy calibration is reliable across the full fitting region.

Those assumptions matter unevenly. The fitted temperature is mostly controlled
by the slope of the tail and is therefore relatively insensitive to an overall
scale factor. `Delta_mu`, by contrast, depends on the intercept and therefore on
absolute calibration.

### 4.2 Excitation intensities

The laser intensities are not stored in the PL file. They are supplied as the
array `EXCITATION_INTENSITY_W_CM2` in `hot_carrier/config.py`. The pipeline
checks that the number of intensities matches the number of spectra exactly.

This explicit mapping is scientifically preferable to parsing intensity from a
file name because:

- it makes the mapping auditable,
- it avoids hidden ordering assumptions,
- it prevents accidental unit drift.

### 4.3 PLQY file

The optional PLQY file is resolved in
`hot_carrier.pipeline._resolve_plqy_profiles()`.

The function tries, in order:

1. interpolation versus absorbed-photon scale,
2. interpolation versus excitation intensity,
3. row-aligned assignment,
4. fallback to constant `PLQY_ETA` and `PLQY_ETA_SIGMA`.

For the bundled dataset the code prints a warning that the `phiabs` scale does
not overlap the experimental `phi_abs` scale and therefore falls back to the
row-aligned interpretation:

```text
table:GaAs bulk_PLQY_results.csv (row-aligned)
```

That is the correct conservative choice for the present files. The PLQY table is
labeled in absolute `photons/s`, while the power-balance stage works with
area-normalized photon flux in `cm^-2 s^-1`. Without a beam area or another
conversion factor, direct interpolation in `phiabs` would mix incompatible
quantities.

## 5. Optical Model: From the PL Tail to `T` and `Delta_mu`

### 5.1 Generalized Planck law

The code models the emitted PL intensity as:

$$
I_{\mathrm{PL}}(E) \sim \frac{2E^2}{h^3 c^2} A(E)
\left[\exp\left(\frac{E-\Delta\mu}{k_B T}\right)-1\right]^{-1}
$$

This is the generalized Planck law used throughout the hot-carrier PL
literature. The underlying assumptions are standard:

- the carrier population is close enough to internal quasi-equilibrium that a
  single carrier temperature is meaningful,
- the electron and hole populations can be summarized by quasi-Fermi levels,
- the optical transition matrix elements do not introduce sharp structure inside
  the narrow fit window.

The code does not prove those assumptions from first principles. It takes them
as the working model and then checks whether the measured tail is in fact close
to linear after the standard transformation. The very high `R^2` values reported
for all 22 fits are evidence that this reduced description is at least
self-consistent on the fitted interval.

### 5.2 Why only the high-energy tail is fitted

The code does not fit the PL peak or the near-gap shoulder. It fits only the
high-energy tail, for two independent reasons.

First, the tail is the part of the spectrum most directly controlled by the
Boltzmann factor. Near the peak, reabsorption, detailed spectral structure of
`A(E)`, and small deviations from the simple asymptotic form matter more.

Second, in the tail the Bose-Einstein denominator simplifies:

$$
\left[\exp\left(\frac{E-\Delta\mu}{k_B T}\right)-1\right]^{-1}
\approx
\exp\left[-\frac{E-\Delta\mu}{k_B T}\right]
$$

The present fits are comfortably inside that asymptotic regime. Across the 22
selected windows, the lower fit bound satisfies:

$$
\frac{E_{\mathrm{fit,min}}-\Delta\mu}{k_B T} = 4.85 \text{ to } 13.37
$$

So even the least favorable fitted point is almost five thermal energies above
`Delta_mu`. That is exactly the regime where the `-1` term is negligible and a
straight-line fit is physically justified.

### 5.3 Constant high-energy absorptivity

After the high-energy approximation, the code assumes that absorptivity is
locally constant over the fit window:

$$
I_{\mathrm{PL}}(E) \sim \frac{2E^2}{h^3 c^2} A_0
\exp\left[-\frac{E-\Delta\mu}{k_B T}\right]
$$

This is the most consequential optical simplification in the repository.

Why the code makes this choice:

- there is no measured spectral absorptivity curve bundled with the repository,
- a constant `A0` turns the optical inversion into a transparent linear problem,
- for a relatively thick `950 nm` GaAs layer, the absorptivity in the upper
  near-gap region is expected to vary much more slowly than in the immediate
  Urbach-tail region.

Why that is reasonable here:

- the PL peaks lie at `1.425-1.435 eV`, very close to the configured room
  temperature gap `E_g = 1.424 eV`,
- the selected fit windows start `30-125 meV` above the PL peak, so they avoid
  the most strongly curved near-edge region,
- the lab note bundled with the repository states that an independent OptiPV
  optical simulation gave a high-energy absorptivity interval `0.459-0.555`
  over `1.50-1.70 eV` for the `950 nm` GaAs layer.

Why it is still a source of systematic error:

- the actual selected windows span `1.4586-1.7662 eV`,
- the supplied `A0` interval was estimated only on `1.50-1.70 eV`,
- the code therefore extrapolates the flat-`A0` assumption slightly below and
  above the interval from which `A0` was originally estimated.

This approximation mostly affects `Delta_mu`, not `T`. A multiplicative change
in `A0` shifts the intercept. It does not change the slope of the linearized
tail, so the temperature is much more robust than the inferred carrier density.

### 5.4 Linearization used in the code

The fitted quantity is:

$$
y(E) = \ln\left[\frac{h^3 c^2}{2E^2} I_{\mathrm{PL}}(E)\right] = mE + b
$$

This transformation removes the deterministic `E^2` prefactor so that the
remaining slope is controlled by the exponential tail itself.

Implementation detail: the energy array is read in `eV`, but the regression is
performed in joules. That is why the code can recover temperature directly from:

$$
T = -\frac{1}{k_B m}
$$

The intercept first gives:

$$
\Delta\mu_{\mathrm{eff}} = \frac{b k_B T}{e}
$$

and only then subtracts the absorptivity contribution:

$$
\Delta\mu = \Delta\mu_{\mathrm{eff}} - \frac{k_B T}{e}\ln(A_0)
$$

The distinction is deliberate. `Delta_mu_eff` is what the raw line intercept
contains. `Delta_mu` is the intercept after imposing the chosen `A0`.

The important plotting point is that the **actual least-squares regression is
performed only on this linearized quantity** `y(E)` versus `E`. In each
`outputs/fits/fit_spectrum_XX.png`:

- the **bottom panel** is the object that is actually fit,
- the **top panel** shows the raw GPL spectrum together with the model obtained
  by back-transforming that lower-panel regression.

So the dashed red curve in the top panel should not be read as "a line was fit
directly on the log-intensity GPL plot". It was not. The reported `R^2` is the
quality of the lower-panel linear regression in
`ln[(h^3 c^2 / 2E^2) I_PL]` versus photon energy.

### 5.5 Why temperature is more trustworthy than density

The code reports both temperature and density-related quantities, but they do
not have equal epistemic status.

- `T` comes from the slope of a nearly straight line over `49-71` points with
  `R^2 > 0.9995`.
- `Delta_mu` comes from the intercept of that line and therefore inherits every
  multiplicative calibration uncertainty, including the assumed `A0`.
- `mu_e`, `mu_h`, and `n` are then reconstructed from `T` and `Delta_mu`, so
  they inherit the intercept sensitivity as well.

The practical consequence is that the code can make a fairly strong statement
about the existence of hot carriers and a weaker statement about the exact
carrier density unless `A(E)` is characterized more completely.

## 6. How the Fitting Window Is Chosen and Why

### 6.1 Search domain

The automatic window search uses:

- `WINDOW_SEARCH_MIN_EV = 1.45 eV`
- `WINDOW_SEARCH_MAX_EV = 1.77 eV`
- `WINDOW_PEAK_OFFSET_EV = 0.0 eV`

The peaks in the present dataset lie between `1.425` and `1.435 eV`, so the
effective search always begins above the PL peak even though the explicit peak
offset is set to zero. This is a sensible choice:

- the lower bound sits only `15-25 meV` above the peak, which keeps the search
  close enough to the onset of the tail that the signal remains strong,
- but it is still above the most strongly curved near-peak region where a
  constant `A(E)` is least defensible,
- the upper bound reaches the top of the measured energy axis, so the search is
  free to use as much of the clean tail as the data justify.

### 6.2 Candidate-window acceptance criteria

Every contiguous candidate window inside the search domain is tested. The code
keeps only windows that satisfy:

- positive intensity at every point,
- at least `WINDOW_MIN_POINTS = 18` points,
- negative fitted slope,
- `R^2 >= 0.995`,
- `150 K <= T <= 1200 K`.

Each threshold has a physical role:

- Positive intensities are required because the logarithm is taken explicitly.
- A minimum of 18 points prevents a visually straight but statistically fragile
  3- or 4-point line from dominating the fit.
- The slope must be negative because a hot PL tail should decay with energy.
- The high `R^2` threshold enforces that the selected interval is genuinely
  close to exponential.
- The broad temperature bounds are not meant to encode prior belief about the
  sample. They are only a guardrail against mathematically valid but physically
  absurd line fits.

### 6.3 Why AICc is used instead of picking the widest or highest-`R^2` window

Among the acceptable windows, the code chooses the one with the smallest AICc:

$$
\begin{aligned}
\mathrm{AIC} &= N \ln(\mathrm{RSS}/N) + 2k \\
\mathrm{AIC}_c &= \mathrm{AIC} + \frac{2k(k+1)}{N-k-1}
\end{aligned}
$$

with `k = 2` parameters.

This is a better choice than maximizing `R^2` alone because:

- on a narrow tail, many windows can have `R^2` values that are all extremely
  close to one,
- a trivially short window can fit almost perfectly but give an unstable slope,
- AICc balances residual quality against sample size and therefore discourages
  overfitting by too-short windows.

In the present outputs every spectrum uses the mode:

```text
auto_peak_offset|aicc
```

So no spectrum had to fall back to a manual or degenerate window.

### 6.4 What the chosen windows look like on this dataset

The selected windows are not arbitrary. Across the 22 spectra, the code chose:

- lower bounds `1.4586-1.5537 eV`,
- upper bounds `1.7031-1.7662 eV`,
- `49-71` fitted points per spectrum.

Relative to the PL peak, the selected windows begin `30-125 meV` above the peak
and extend `278-338 meV` above it. That is exactly what one would expect if the
algorithm is locating a region where the spectrum has already crossed over to a
clean exponential tail.

### 6.5 Fit-window uncertainty and why it is treated separately

The final answer should not depend too strongly on one exact window boundary.
For that reason the code computes a second ensemble over plausible neighboring
windows using the looser settings:

- `FIT_RANGE_SCAN_MIN_POINTS = 12`
- `FIT_RANGE_SCAN_MIN_R2 = 0.99`

This looser ensemble is used only for uncertainty estimation, not for the
central fit. The reason is straightforward:

- the best-fit window should be strict,
- the uncertainty analysis should be broad enough to probe window sensitivity.

For the current dataset, each spectrum retains roughly `2071-2211` acceptable
windows in this ensemble. The code then AICc-weights them and reports the RMS
spread of the inferred parameters as the `range` uncertainty.

That is a principled replacement for the common but poorly documented habit of
trying "a few other windows by hand."

### 6.6 Total uncertainty model

For each reported parameter, the code combines three terms:

$$
\sigma_{\mathrm{total}}^2 = \sigma_{\chi^2}^2 + \sigma_{\mathrm{range}}^2 + \sigma_{A_0}^2
$$

Their meanings are different:

- `chi2`: local statistical uncertainty of the chosen line fit,
- `range`: sensitivity to the fact that the fit window is a modelling choice,
- `A0`: systematic uncertainty from the assumed high-energy absorptivity.

This separation matters physically. A fit can have tiny line-fit covariance and
still be uncertain because another plausible window gives a slightly different
slope or because `A0` is not perfectly known.

## 7. Reconstructing `mu_e`, `mu_h`, and `n`

### 7.1 Common thermodynamic model

The code assumes parabolic conduction and valence bands with effective masses:

- `m_e* = 0.067 m0`
- `m_h* = 0.50 m0`

and a fixed band gap:

- `E_g = 1.424 eV`

The reported chemical potentials are referenced to mid-gap, so the band edges
are at `+E_g/2` and `-E_g/2`.

Why this is reasonable:

- the fitted PL tail is extracted close to the direct gap, where the effective
  mass approximation is standard,
- the extracted temperatures remain modest, `299-378 K`, so the relevant
  thermal energies are only `26-33 meV`,
- the PL peaks are themselves clustered near `1.424-1.435 eV`, consistent with
  a room-temperature GaAs gap.

What is left out:

- temperature dependence of `E_g`,
- band nonparabolicity,
- light-hole and heavy-hole substructure,
- doping-induced asymmetry.

The code therefore reconstructs a clean reduced state, not the last word on the
true many-body band structure.

### 7.2 Why charge neutrality is imposed

The inversion from `Delta_mu` to separate electron and hole chemical potentials
is underdetermined unless one adds another condition. The code uses:

$$
n = p
$$

This is the usual steady-state equal-injection closure for optically excited,
undriven bulk material. It is reasonable because the repository does not model a
built-in electric field, a current-extracting junction, or strongly asymmetric
doping. Each absorbed photon is therefore treated as creating one electron-hole
pair, and the steady state remains close to charge neutrality.

If the sample had strong doping, carrier extraction, or spatial separation, this
closure would have to be generalized.

### 7.3 Maxwell-Boltzmann reconstruction

Under MB statistics, the code computes the effective densities of states:

$$
\begin{aligned}
N_c(T) &= 2\left(\frac{m_e^* k_B T}{2\pi \hbar^2}\right)^{3/2} \\
N_v(T) &= 2\left(\frac{m_h^* k_B T}{2\pi \hbar^2}\right)^{3/2}
\end{aligned}
$$

and then solves analytically:

$$
\begin{aligned}
\mu_e &= \frac{1}{2}\left[\Delta\mu - \frac{k_B T}{e}\ln\left(\frac{N_c}{N_v}\right)\right] \\
\mu_h &= \frac{1}{2}\left[\Delta\mu + \frac{k_B T}{e}\ln\left(\frac{N_c}{N_v}\right)\right] \\
n &= N_c \exp\left[\frac{(\mu_e - E_g/2)e}{k_B T}\right]
\end{aligned}
$$

The attraction of the MB version is not just speed. It is also interpretability.
The mass asymmetry enters through `ln(N_c/N_v)`, so one can see directly how the
heavier hole mass pushes `mu_h` and `mu_e` apart even at fixed `Delta_mu`.

### 7.4 Fermi-Dirac reconstruction

The code also solves the full charge-neutral Fermi-Dirac problem with the
complete `F_{1/2}` integral and a bisection search.

This is not a cosmetic addition. It addresses a real issue in the present data:

- the experimental reduced variable
  $x = (\Delta\mu - E_g)/(k_B T)$ reaches $-0.94$ at the high-intensity end,
- the repository's own MB-validity scan places a conservative `5 %` boundary at
  $x^* \approx -2.33$,
- therefore the most strongly pumped spectra are no longer safely in the MB
  regime.

The effect is visible in the exported densities:

- MB density reaches `1.70e18 cm^-3`,
- FD density reaches `1.20e18 cm^-3`.

That direction of change is physically expected. Once degeneracy matters, the
Pauli principle suppresses occupation growth compared with the classical MB
formula, so MB tends to overestimate density at fixed `Delta_mu`.

### 7.5 Why both MB and FD are kept

The code exports both reconstructions for a reason:

- MB is still useful as a transparent low-density baseline,
- FD is safer once the system approaches degeneracy,
- the disagreement between the two is itself a diagnostic of where classical
  closure is no longer adequate.

The current repository therefore uses FD by default where a single "best"
per-carrier quantity is needed, but it keeps the MB companion values visible so
that the statistical closure remains falsifiable.

## 8. MB Validity Diagnostic for the Photon Occupation

The repository contains a second MB check that is conceptually different from
the MB-vs-FD carrier-density comparison above. Here the question is:

> When is it still safe to replace the Bose-Einstein photon occupation in the
> generalized Planck law by its first Boltzmann term?

To isolate that issue cleanly, the code uses a step absorber:

$$
A(E) = A_0 \, \Theta(E - E_g)
$$

and compares the integrated generalized Planck law computed with:

- the exact Bose-Einstein occupation,
- the MB approximation.

Why a step absorber is used here:

- it removes material-specific spectral structure from the diagnostic,
- it makes the result depend only on the reduced variable
  `x = (Delta_mu - E_g)/(k_B T)`,
- it therefore answers a narrow but important question: when does the photon
  occupation approximation itself fail?

This diagnostic is intentionally not a full sample model. It is a conservative
closure test.

For the bundled data:

- the conservative `5 %` threshold is $x^* \approx -2.33$,
- the experimental points span $x = -11.76$ to $-0.94$,
- `9` of the `22` spectra lie beyond that limit.

So even if MB remains useful for low-intensity intuition, the high-intensity end
of this dataset should not be interpreted as safely classical.

## 9. Power Balance: From Excitation Intensity to Thermalized Power

### 9.1 Algebra implemented in the code

The power-balance stage in `hot_carrier.analysis.compute_power_balance_table()`
uses:

$$
\begin{aligned}
P_{\mathrm{abs}} &= A_{\mathrm{laser}} P_{\mathrm{exc}} \\
\phi_{\mathrm{abs}} &= \frac{P_{\mathrm{abs}}}{E_{\mathrm{laser}}} \\
\phi_{\mathrm{rad}} &= \eta \phi_{\mathrm{abs}} \\
\phi_{\mathrm{nonrad}} &= (1-\eta)\phi_{\mathrm{abs}}
\end{aligned}
$$

and then:

$$
\begin{aligned}
P_{\mathrm{rec}} &= \phi_{\mathrm{nonrad}} E_{\mathrm{nonrad}}
                 + \phi_{\mathrm{rad}} E_{\mathrm{rad}} \\
P_{\mathrm{th}} &= P_{\mathrm{abs}} - P_{\mathrm{rec}}
\end{aligned}
$$

Finally it converts `W cm^-2` to `W cm^-3` by dividing by the active-layer
thickness `d = 950 nm`.

### 9.2 Why this compact model is used

This is a bookkeeping model, not a transport simulation. Its purpose is to
translate the measured optical state and the supplied PLQY into the quantity
needed for the Tsai comparison:

$$
P_{\mathrm{th,exp}}
$$

The code does not resolve spatial gradients, diffusion, photon recycling, or a
full microscopic recombination spectrum. It uses the smallest model that still
enforces both carrier-flux and energy balance.

### 9.3 Why the code uses the nonradiative recombination-energy model

The code assumes:

$$
E_{\mathrm{nonrad}} = E_g + 3 k_B T
$$

This is a physically motivated pair-energy estimate. In a 3D parabolic band,
the mean kinetic energy of a classical carrier is `3/2 k_B T`. For a
nonradiative electron-hole recombination event, both carrier kinetic energies
are dumped back into the lattice along with the band-gap energy. That yields:

$$
E_g + \frac{3}{2} k_B T + \frac{3}{2} k_B T = E_g + 3 k_B T
$$

The assumption is therefore not arbitrary. It is the simplest thermal average
consistent with the same parabolic-band picture already used elsewhere in the
repository.

### 9.4 Why the code uses the radiative recombination-energy model

The code assumes:

$$
E_{\mathrm{rad}} = E_g + k_B T
$$

This is a compact near-edge estimate for the average emitted photon energy in
radiative recombination. The key idea is that radiative recombination does not
return the full pair thermal energy to the lattice. Part of that excess energy
leaves as photon energy, and the spectral average of that emitted photon sits
slightly above `E_g`.

The code chooses the `+ k_B T` form because it is a standard near-edge thermal
scale and keeps the power balance analytically transparent. It should be read as
a spectral average approximation, not as a line-by-line radiative transfer
calculation.

The good news is that, in the bundled dataset, PLQY is low enough that this
approximation does not dominate the energy balance. The radiative fraction of
absorbed power is only `0.14-3.61 %`, whereas the nonradiative fraction is
`60.8-64.3 %`. So the power balance is much more sensitive to `A_laser`, PLQY,
and thickness than to the precise sub-gap correction on `E_rad`.

### 9.5 Why the power is converted to a volume density

Tsai's `du/dt` is an energy-density rate, not an area-normalized flux. That is
why the code divides `P_th` by the active-layer thickness before comparing it to
the Tsai cooling kernel.

This step is physically necessary, not cosmetic. A direct comparison between
`W cm^-2` from the experiment and `W cm^-3` from the microscopic model would be
dimensionally wrong.

### 9.6 What the power-balance stage does not claim

The thermalized power reported here is model-derived. It is not a direct
calorimetric measurement. It inherits assumptions about:

- laser absorptivity at `532 nm`,
- PLQY assignment,
- active-layer thickness,
- average recombination energies,
- the single-temperature electronic picture itself.

It is the best estimate that this reduced model can produce, not an exact
observable.

In particular, the workflow does not solve for pump-induced lattice heating or a
spatially varying lattice temperature. The lattice is treated as a fixed
background reference, so the reported `P_th` quantifies carrier-side energy
dissipation within that approximation, not a self-consistent lattice heat-up.

## 10. Why the Tsai Stage Is Electron-Only

The Tsai workflow in this repository models intraband cooling by electrons only.
That deserves explicit justification.

For a direct optical transition in parabolic bands, the electron and hole are
created at the same `|k|`, so their excess kinetic energies split as:

$$  
\frac{E_{\mathrm{excess},\mathrm{e}}}{E_{\mathrm{excess},\mathrm{total}}}
= \frac{1/m_e^{*}}{1/m_e^{*} + 1/m_h^{*}}
= \frac{m_h^{*}}{m_e^{*} + m_h^{*}}
$$

Using the masses configured in this repository:

- `m_e^{*} = 0.067`
- `m_h^{*} = 0.50`

gives:

- electron share $\approx 0.882$
- hole share $\approx 0.118$

So, in this reduced picture, about `88 %` of the excess kinetic energy is
initially assigned to electrons. That is the physical reason the code treats
electron cooling as the dominant hot-carrier channel.

This does **not** mean the code rigorously subtracts the hole contribution from
the measured thermalized power. It does not. It means the repository assumes
that the experimental `P_th` is dominated by the electron channel strongly
enough that an electron-only Tsai comparison is informative.

That is a defensible first approximation in GaAs. It is not a proof that holes
are negligible.

## 11. Tsai Eq. 41 Plus Eq. 48 Cooling Workflow

### 11.1 What is and is not implemented

The repository does not reproduce Tsai's full photovoltaic device model. It
implements the electron intraband cooling kernel and uses it as a consistency
check against the experimentally inferred thermalized power.

The steady-state identification is:

$$
P_{\mathrm{th,exp}} \approx -\left(\frac{du}{dt}\right)_{\mathrm{intra}}
$$

This is narrower than solving Tsai Eq. 3 together with a complete transport and
`J-V` model. The comparison is still useful, but the interpretation must stay at
that level.

Two points must be stated explicitly because they are easy to blur together:

- the Tsai stage is **not** a free-parameter fit,
- the Tsai stage is **not** driven by raw spectra alone.

Once `hot_carrier/config.py` is fixed, the Tsai workflow does **not** adjust
`tau_LO`, dielectric constants, effective masses, screening settings, or grid
bounds to force agreement with experiment. It deterministically evaluates the
chosen model over a state-temperature grid and then compares that model to the
experimentally inferred state.

The Tsai comparison therefore mixes three kinds of inputs:

1. Quantities inferred from experiment **before** the Tsai stage:
   - `Delta_mu_exp` and `T_exp`, obtained from the optical GPL-tail fit,
   - `P_th_exp`, obtained from the absorbed-minus-recombination power balance.
2. Fixed model parameters chosen by the user or taken from external material
   inputs:
   - `TSAI_LO_PHONON_LIFETIME_PS`,
   - `TSAI_LO_PHONON_ENERGY_EV`,
   - `TSAI_EPSILON_INF`, `TSAI_EPSILON_STATIC`,
   - `TSAI_LATTICE_TEMPERATURE_K`,
   - `EG_EV`, `M_E_EFF`,
   - `TSAI_SCREENING_MODEL`, `TSAI_USE_STATIC_SCREENING`,
   - `TSAI_DELTA_MU_CARRIER_STATISTICS`.
3. Numerical construction choices:
   - whether the inverse map is built on `Delta_mu` or `mu_e`,
   - the `q` integration limits and resolution,
   - the temperature and state-axis grid bounds,
   - interpolation in `log10(P_th)`.

So the Tsai stage is best described as a **no-fit, physics-based comparison
conditioned on experimentally inferred inputs and on a chosen parameter set**.
It is not a parameter-free derivation from raw data plus fundamental constants.

### 11.2 Equations used

The implemented hierarchy is:

$$
\text{Eq. 34} \rightarrow \text{Eq. 41} \rightarrow \text{Eq. 48}
$$

with the steady-state hot-phonon relations between them.

The code keeps the screened polar LO interaction, computes the mode-resolved
electron-to-LO emission time `tau_c-LO^q`, and then integrates the net
electron-LO energy transfer rate over `q`.

### 11.3 Static screening and why it is acceptable here

The screening model is static Thomas-Fermi:

$$
\varepsilon(q,0) = \varepsilon_0 \left(1 + \frac{q_s^2}{q^2}\right)
$$

with a default screening closure:

```text
TSAI_SCREENING_MODEL = "mb"
```

This is a pragmatic choice:

- it matches the analytic structure used in Tsai's derivation,
- it is numerically stable,
- it captures the main trend that stronger carrier compressibility leads to
  stronger screening and weaker electron-phonon coupling.

The repository also allows a finite-difference FD compressibility
(`"fd_fdiff"`), but the default stays MB because it is simpler and closer to the
paper's compact form.

This creates an intentional hybrid model in the default settings:

- `Delta_mu -> mu_e` reconstruction for the Tsai grid uses FD,
- screening itself still uses MB.

That is not a coding mistake. It reflects two separate modelling layers:

- one layer reconstructs the carrier state from the optical fit,
- the other layer approximates the screening response entering the Fröhlich
  matrix element.

### 11.4 Why the `q` integration is done on a geometric grid

The Fröhlich interaction scales like `1/q^2`, screening modifies the small-`q`
region strongly, and the integral spans several decades in `q`. A geometric grid
therefore makes more sense than a linear grid because it resolves the low-`q`
structure without wasting most points at large `q`.

The current settings are:

- `TSAI_Q_MIN_CM1 = 3e4`
- `TSAI_Q_MAX_CM1 = 1e8`
- `TSAI_Q_POINTS = 520`

`q_min` is kept strictly positive because the bare Fröhlich term diverges at
`q = 0`; the exact zero mode is neither numerically stable nor physically the
right object for a finite-bandwidth calculation.

### 11.5 Why the default primary axis is `Delta_mu`

The inverse Tsai workflow can be built on either `mu_e` or `Delta_mu`. The
default is:

```text
TSAI_PRIMARY_INPUT = "delta_mu"
```

This reduces circularity. If `mu_e` were used directly, the Tsai model would be
fed a quantity that had already been reconstructed from the experimental `T`.
Using `Delta_mu` instead leaves the Tsai stage to rebuild `mu_e(Delta_mu, T)`
internally on each trial temperature.

The resulting map is:

$$
\Delta\mu,\; T \rightarrow \mu_e,\; n \rightarrow q_s
\rightarrow \tau_{c\text{-}\mathrm{LO}}^q \rightarrow \left(\frac{du}{dt}\right)_{\mathrm{intra}}
$$

This is not fully model-independent, because the `Delta_mu -> mu_e` conversion
still depends on the assumed band structure and carrier statistics. But it is
less circular than supplying the experimental `mu_e` directly.

### 11.6 Why the inverse map is built in `log10(P_th)`

The modeled thermalized power spans nearly three decades across the current
dataset. Interpolating directly in linear `P_th` would over-resolve the
high-power region and under-resolve the low-power region. The inverse map is
therefore built in `log10(P_th)`.

That is a data-processing choice with a clear physical rationale: the relevant
comparison is multiplicative over a wide dynamic range, not additive over a
narrow one.

### 11.7 How `Q` is extracted from the `P_th` vs `\Delta T` figure

The `outputs/tsai_temperature_rise_vs_pth_density.png` figure is displayed as
carrier temperature rise versus thermalized power density with a logarithmic
`P_th` axis. The `Q` extraction is **not** obtained by fitting a straight line
directly in those displayed axes.

Instead, the code linearizes the Tsai-style thermalization relation as:

$$
P_{\mathrm{th}}\exp\left(\frac{E_{\mathrm{LO}}}{k_B T_c}\right) = Q(T_c-T_L) + b
= Q\,\Delta T + b
$$

with:

- `T_c = T_L + Delta T`,
- `Q` as the thermalization coefficient,
- `b` as a free intercept rather than forcing the line through the origin.

The regression is performed on the linearized quantity
`P_th exp(E_LO / k_B T_c)` versus `Delta T`. The curve shown on
`outputs/tsai_temperature_rise_vs_pth_density.png` is the back-transformed
result drawn on the original `Delta T` versus `P_th` axes, so it does not
appear as a straight line in the displayed figure.

That regression is a compact diagnostic only. It does **not** feed back into the
Tsai forward map, does **not** tune `tau_LO` or any other Tsai parameter, and
does **not** affect the simulated temperatures reported by the inverse-map
comparison.

With the present defaults and bundled dataset, the fitted `Q` values are:

- experimental points: `Q ~= 1.22 x 10^7 W cm^-3 K^-1`, `R^2 ~= 0.978`
- Tsai FD prediction: `Q ~= 1.26 x 10^7 W cm^-3 K^-1`, `R^2 ~= 0.999`
- Tsai MB prediction: `Q ~= 1.19 x 10^7 W cm^-3 K^-1`, `R^2 ~= 0.999`

These `Q` values are useful compact diagnostics of how steeply the thermalized
power rises with carrier temperature under the Tsai-style LO-phonon activation
factor. They are not a substitute for the full inverse-map comparison.

### 11.8 What the current Tsai comparison means

The code:

1. builds a forward map `(state, T) -> P_th_model`,
2. numerically inverts it to `(state, P_th) -> T`,
3. evaluates that inverse map at the experimental `(Delta_mu_exp, P_th_exp)`,
4. compares the predicted temperature with the optical temperature.

With the present defaults:

- `TSAI_LO_PHONON_LIFETIME_PS = 16.0`
- `TSAI_PRIMARY_INPUT = "delta_mu"`
- `TSAI_DELTA_MU_CARRIER_STATISTICS = "fd"`
- `TSAI_SCREENING_MODEL = "mb"`

the comparison yields:

- FD default closure: MAE $\approx 3.31\ \mathrm{K}$, bias $\approx -1.76\ \mathrm{K}$
- MB comparison closure: MAE $\approx 3.15\ \mathrm{K}$, bias $\approx -0.68\ \mathrm{K}$

This is reasonably good agreement on a temperature rise that spans only a few
tens of kelvin. The correct interpretation is:

- the experimentally inferred thermalized-power density is broadly compatible
  with a Tsai-type electron-LO cooling law under the chosen parameters.

The incorrect interpretations would be:

- that the full Tsai paper has been reproduced,
- that the LO-phonon lifetime has been uniquely identified,
- that hole cooling is negligible by proof rather than by approximation,
- that the optical power-balance model is now independently validated in all its
  details.

Agreement here is a consistency test, not a uniqueness theorem.

## 12. Outputs and How to Read Them

The main outputs are:

- `outputs/fit_results.csv`: central per-spectrum table containing fitted
  optical parameters, MB and FD state variables, uncertainties, and power
  channels.
- `outputs/all_spectra_logscale.png`: raw spectra across all excitation
  intensities.
- `outputs/fits/fit_spectrum_XX.png`: two-panel optical-fit diagnostic; the
  lower panel is the actual linear regression on the linearized GPL tail, while
  the upper panel shows the back-transformed model on the raw PL scale.
- `outputs/parameters_vs_intensity.png`: extracted state variables versus pump
  intensity.
- `outputs/thermalized_power_diagnostics.png`: thermalized-power diagnostics and
  per-carrier cooling quantities.
- `outputs/recombination_channel_contributions.png`: radiative versus
  nonradiative channel comparison.
- `outputs/tsai_temperature_rise_vs_pth_density.png`: experimental and
  Tsai-predicted `Delta T` versus `P_th`, including back-transformed `Q`-fit
  curves and annotated `Q` values from the linearized relation above.
- `outputs/mb_validity_limit.png`, `outputs/mb_validity_scan.csv`,
  `outputs/mb_validity_limits.csv`: the photon-occupation MB diagnostic.
- `outputs/tsai_forward_stateT_to_pth.csv`,
  `outputs/tsai_inverse_pth_state_to_temperature.csv`,
  `outputs/tsai_du_dt_samples_at_experimental_state.csv`,
  `outputs/tsai_temperature_comparison.csv`: the Tsai forward and inverse maps
  plus the experimental comparison.

When reading `outputs/thermalized_power_diagnostics.png`, the main caution is
panel `(c)`, the per-carrier cooling-rate plot `P_th / n`. In the present
interpretation, the fact that `P_th / n` first decreases, reaches a minimum,
and then increases again should not be over-read as a robust intrinsic cooling
signature. The more likely explanation is uncertainty in the GaAs sample doping,
which is not independently pinned down in the current workflow and can shift the
reconstructed carrier density. The workflow also does **not** include
self-consistent lattice heating, so these diagnostics should be read as carrier
heating above a fixed lattice background rather than a full carrier-lattice
thermal model.

The most important columns in `fit_results.csv` are:

- `temperature_k`, `qfls_ev`
- `mu_e_ev`, `mu_h_ev`, `carrier_density_cm3`
- `mu_e_fd_ev`, `mu_h_fd_ev`, `carrier_density_fd_cm3`
- `temperature_err_total_k`, `qfls_err_total_ev`,
  `carrier_density_fd_err_total_cm3`
- `absorbed_power_w_cm3`, `recombination_power_w_cm3`,
  `thermalized_power_w_cm3`
- `thermalized_power_per_carrier_mb_ev_s`,
  `thermalized_power_per_carrier_fd_ev_s`
- `fit_min_ev`, `fit_max_ev`, `n_points_fit`, `r2`, `fit_range_samples`

The most important columns in `tsai_temperature_comparison.csv` are:

- `delta_mu_ev`
- `temperature_k_exp`
- `p_th_exp_w_cm3`
- `temperature_sim_k`
- `temperature_error_k`
- the MB and FD companion columns for `mu_e` and carrier density

## 13. Configuration Values That Matter Most

Most settings can be grouped by scientific role.

### 13.1 Optical fit

- `AUTO_SELECT_FIT_WINDOW`
- `WINDOW_SEARCH_MIN_EV`, `WINDOW_SEARCH_MAX_EV`
- `WINDOW_MIN_POINTS`, `WINDOW_MIN_R2`
- `FIT_RANGE_SCAN_MIN_POINTS`, `FIT_RANGE_SCAN_MIN_R2`

These control how aggressively the code searches for and stress-tests the
exponential tail.

### 13.2 High-energy absorptivity

- `A0_HIGH_ENERGY_MIN = 0.459`
- `A0_HIGH_ENERGY_MAX = 0.555`
- `ASSUMED_A0 = 0.507`
- `A0_SIGMA = (A0_max - A0_min) / sqrt(12)` for the default `"uniform"` model

This is the most important optical systematic because it shifts `Delta_mu`
directly.

The use of a uniform uncertainty model is justified by the information actually
available in the repository: only an interval is given, not a best-resolved
spectral profile or a reason to prefer one value inside that interval over
another.

### 13.3 Carrier thermodynamics

- `EG_EV = 1.424`
- `M_E_EFF = 0.067`
- `M_H_EFF = 0.50`

These parameters govern the `Delta_mu -> (mu_e, mu_h, n)` inversion and also
enter the physical argument for electron-dominated excess-energy partitioning.

### 13.4 Power balance

- `LASER_WAVELENGTH_NM = 532.0`
- `ABSORPTIVITY_AT_LASER = 0.625`
- `PLQY_RESULTS_CSV = "GaAs bulk_PLQY_results.csv"`
- `ACTIVE_LAYER_THICKNESS_NM = 950.0`

These settings determine the experimental `P_th` that the Tsai stage tries to
match.

### 13.5 Tsai workflow

The Tsai comparison depends on a set of fixed modelling choices. These are
**chosen inputs**, not parameters extracted by the Tsai workflow:

- `TSAI_LATTICE_TEMPERATURE_K = 298.15`: reference lattice temperature.
- `TSAI_LO_PHONON_ENERGY_EV = 0.03536`: LO-phonon energy entering the
  occupation factors and Eq. 48.
- `TSAI_LO_PHONON_LIFETIME_PS = 16.0`: effective hot-phonon lifetime. This is a
  user-specified parameter in the present workflow, not an output uniquely
  identified from the bundled dataset.
- `TSAI_EPSILON_INF = 10.89`, `TSAI_EPSILON_STATIC = 12.90`: dielectric inputs
  entering the Fröhlich coupling and screening.
- `EG_EV = 1.424`, `M_E_EFF = 0.067`: GaAs band and electron effective-mass
  inputs used in the state reconstruction and cooling kernel.
- `TSAI_PRIMARY_INPUT = "delta_mu"`: the inverse map is parameterized by
  `Delta_mu` rather than directly by `mu_e`.
- `TSAI_DELTA_MU_CARRIER_STATISTICS = "fd"`: `Delta_mu -> mu_e` reconstruction
  inside the Tsai grid uses Fermi-Dirac carrier statistics.
- `TSAI_USE_STATIC_SCREENING = True`,
  `TSAI_SCREENING_MODEL = "mb"`: static Thomas-Fermi screening with an MB
  compressibility closure by default.
- `TSAI_Q_MIN_CM1 = 3e4`, `TSAI_Q_MAX_CM1 = 1e8`, `TSAI_Q_POINTS = 520`: `q`
  integration domain and resolution for Eq. 48.
- `TSAI_T_GRID_MIN_K = 280.0`, `TSAI_T_GRID_MAX_K = 950.0`,
  `TSAI_T_GRID_POINTS = 135`: temperature grid for the forward map.
- `TSAI_DELTA_MU_GRID_MIN_EV`, `TSAI_DELTA_MU_GRID_MAX_EV`,
  `TSAI_DELTA_MU_GRID_MARGIN_EV`, `TSAI_DELTA_MU_GRID_POINTS`: state-axis setup
  when `TSAI_PRIMARY_INPUT = "delta_mu"`.
- `TSAI_MU_E_GRID_MIN_EV`, `TSAI_MU_E_GRID_MAX_EV`,
  `TSAI_MU_E_GRID_MARGIN_EV`, `TSAI_MU_E_GRID_POINTS`: alternative state-axis
  setup when `TSAI_PRIMARY_INPUT = "mu_e"`.
- `TSAI_PTH_INVERSE_POINTS = 240`: sampling density for the inverse
  `T(state, P_th)` map in `log10(P_th)`.

The Tsai stage also depends indirectly on earlier modelling choices because the
experimental `P_th` passed into it is not a raw measured channel. It comes from
the power-balance model, which uses:

- laser absorptivity at the pump wavelength,
- PLQY values or the PLQY table mapping,
- active-layer thickness,
- average radiative and nonradiative recombination-energy assumptions.

So even before the Tsai kernel is evaluated, the "experimental" comparison
target is already a model-derived quantity.

Changing any of the fixed Tsai settings above can change the predicted
temperature even if the underlying spectra are unchanged. Agreement between the
Tsai map and experiment should therefore always be read as agreement under a
specific chosen parameterization, not as a parameter-free proof.

## 14. Limitations That Matter Scientifically

The current workflow is coherent, but it remains a reduced model. The most
important limitations are:

1. `A(E)` is treated as constant in the fit window. This is the dominant optical
   simplification.
2. `E_g` is fixed at `1.424 eV` rather than made temperature dependent.
3. The carrier-state reconstruction uses parabolic bands and charge neutrality.
4. The MB-validity scan uses a step absorber to isolate photon-occupation error,
   so it is deliberately generic rather than sample-specific.
5. The power-balance stage uses average recombination energies, not a spectral
   recombination integral.
6. Pump-induced lattice heating is not included; `TSAI_LATTICE_TEMPERATURE_K`
   is treated as a fixed background rather than solved self-consistently.
7. The Tsai comparison is electron-only and treats `tau_LO` as a user-specified
   effective input parameter rather than an output uniquely identified by the
   dataset.
8. Screening is static Thomas-Fermi rather than a full dynamic dielectric
   treatment.
9. The Tsai stage currently has no propagated uncertainty band, so the reported
   temperature comparison does not quantify sensitivity to the chosen Tsai
   parameters.

None of these points invalidates the present analysis. They define its
interpretation domain.

## 15. Highest-Value Improvements

If this repository is meant to become a stronger research tool, the highest
value improvements are:

1. Replace constant `A0` by a measured or simulated spectral `A(E)`.
2. Propagate uncertainty through the Tsai inverse stage.
3. Add a temperature-dependent `E_g(T)` model.
4. Save a full run manifest with configuration, package versions, and git
   commit.
5. Add regression tests on the bundled dataset.
6. Extend the Tsai stage to include hole cooling or at least a controlled
   electron-hole energy partition factor.

## 16. How to Run

The code is designed to be run from the repository root:

```powershell
.\.venv\Scripts\python.exe main.py
```

Or, from a clean environment:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pandas matplotlib scipy
python main.py
```

The run overwrites the generated CSV tables and figures in `outputs/`.

## 17. Local References Used by This Repository

The main local references are:

- [Tsai 2018 hot-carrier model](literature/Tsai_2018_Hot_Carrier_Model.pdf)
- [Vezin 2024 CWPL note](literature/Vezin_2024_CWPL.pdf)
- [Vezin 2025 hot-electron note](literature/Vezin_2025_Hot_Electrons.pdf)
- [Vezin 2025 distinguishing note](literature/Vezin_2025_Distinguishing.pdf)
- [Giteau 2020 hot-carrier reference](literature/Giteau_2020_Hot_Carriers.pdf)
- [Local lab note used to set several analysis choices](Hot%20carrier%20properties%20extraction%20from%20PL%204th%20session.docx)

The lab note is especially relevant for the current implementation because it
documents several choices that the code now formalizes:

- fitting only the high-energy domain,
- treating `A(E)` as `A0` there,
- estimating `A0` from an external optical model,
- reconstructing `mu_e`, `mu_h`, and `n` from `Delta_mu` through
  electroneutrality,
- comparing the experimental thermalized power to a Tsai-style microscopic
  cooling law.
