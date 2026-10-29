# Stochastic Newtonian noise models

This guide outlines the analytical and semi-analytical models implemented in `gragra` for stochastic Newtonian noise (NN) calculations.

Stochastic Newtonian noise arises from ambient random fluctuations—such as seismic surface waves, bulk elastic waves, atmospheric infrasound, and temperature fluctuations. Unlike deterministic sources with prescribed trajectories or geometry, stochastic fields are characterized statistically by power spectral densities (PSDs) or structure functions, and their gravitational coupling produces cross-spectral density (CSD) or acceleration PSD matrices.

## Current status and architecture

The stochastic NN catalogue is currently implemented at the lower numerical tiers:

1. **Analytical kernels (`gragra.analytic`)**: Compute G-free acceleration CSDs and PSDs using closed-form integrals or special-function evaluations.
2. **Observable wrappers (`gragra.observables`)**: Apply physical constants (multiplying by $G^2$ for CSD/PSD outputs), validate array shapes and finite bounds, and return immutable float64 arrays.

High-level integration is deferred: spectral outputs (CSD and PSD matrices) are not yet integrated into the `EvaluationResult` or setup container workflows. Users invoke the observable functions directly with physical parameter values.

## Implemented model catalogue

### 1. Rayleigh wave surface noise (R1)

- **Modules**: `gragra.analytic.stochastic_rayleigh`, `gragra.observables.stochastic_rayleigh`
- **Function**: `rayleigh_isotropic_acceleration_csd`
- **Formulation**: Implements the homogeneous, isotropic Rayleigh wave acceleration CSD for an arbitrary set of test masses at depth, following Harms (2019, Eq. 112) with analytical azimuth integration.
- **Inputs**: Test mass coordinates, medium density $\rho$, compressional velocity $v_p$, shear velocity $v_s$, frequency $f$, and surface vertical displacement PSD $S_{\xi_z}(f)$ in $\mathrm{m^2/Hz}$.
- **Output**: Acceleration CSD tensor $\mathbf{S}_a(f)$ in $(\mathrm{m/s^2})^2/\mathrm{Hz}$.

### 2. Elastic body-wave noise (B1, B2)

- **Modules**: `gragra.analytic.stochastic_bodywave`, `gragra.observables.stochastic_bodywave`
- **Functions**:
  - `fullspace_p_isotropic_acceleration_csd` (B1): Homogeneous and isotropic P-wave field in an unbounded elastic medium.
  - `spherical_cavity_ps_isotropic_acceleration_csd` (B2): Homogeneous, isotropic, mutually incoherent P- and S-wave fields evaluated at the center of a spherical cavity, neglecting boundary scattering.
- **Inputs**: Medium density, wave speeds, frequencies, cavity radius (for B2), and per-component displacement PSDs.

### 3. Atmospheric infrasound noise (A1, A2, A3)

- **Modules**: `gragra.analytic.stochastic_acoustic`, `gragra.observables.stochastic_acoustic`
- **Functions**:
  - `surface_isotropic_acoustic_acceleration_csd` (A1): Isotropic rigid-surface atmospheric infrasound evaluated at ground level.
  - `underground_isotropic_acoustic_acceleration_csd` (A2): The same isotropic infrasound field evaluated at underground depth $h$. *(Note: acoustic formulations are subject to ongoing refinement in parallel development.)*
  - `centered_cavern_isotropic_acoustic_acceleration_csd` (A3): Cavern acoustic mode evaluated at the center of a hemispherical cavern.
- **Inputs**: Air density $\rho_0$, sound speed $c_s$, pressure PSD $S_p(f)$ in $\mathrm{Pa^2/Hz}$, and geometric parameters.

### 4. Atmospheric temperature turbulence (T1)

- **Modules**: `gragra.analytic.stochastic_temperature`, `gragra.observables.stochastic_temperature`
- **Function**: `uniform_advection_temperature_acceleration_psd`
- **Formulation**: Models Newtonian noise from isotropic atmospheric temperature turbulence advected across a line of sight under Taylor's frozen-flow hypothesis with constant wind speed $v$. The turbulence is characterized by a spatial temperature structure function $D_T(r) = c_T^2 r^p$.
- **Inputs**: Mean air density, ambient temperature, structure coefficient $c_T$, exponent $p$, wind speed, minimum approach distance, wind orientation angle, and frequency.

## Random wavefield realizations

In addition to analytic spectral density functions, `gragra.waves.stochastic` provides `stochastic_ps_realization` for generating synthetic random P- and S-wave displacement realizations from drawn wave parameters (`StochasticPSParameters`). This enables time-domain or spatial-domain Monte Carlo studies alongside analytic spectral calculations.
