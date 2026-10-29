# Water Newtonian-noise models API reference

This page documents the analytical Newtonian noise models for water flow, water channels, waterfalls, and surface perturbations.

The functions are divided into two levels:
1. Low-level G-free transfer functions in `gragra.analytic`. These return pure geometric transfer coefficients without multiplying by Newton's gravitational constant $G$.
2. High-level physical observables in `gragra.observables`. These multiply by $G_{\mathrm{SI}}$ and evaluate physical amplitude spectral densities (ASD) and strain ASDs.

## Analytic G-free transfer functions

Low-level G-free transfer functions from `gragra.analytic`.

::: gragra.analytic.water_channel_advection_transfer_g_free

Compute G-free potential, acceleration, and gradient transfer functions for surface displacement advection along a narrow channel.

::: gragra.analytic.water_channel_advection_dc_limit_g_free

Evaluate the exact non-oscillatory DC limit ($f \to 0$) of acceleration and gradient transfer functions for a static surface step.

::: gragra.analytic.water_channel_advection_observables_g_free

Evaluate G-free potential, acceleration, and gradient values packaged in a `GFreeGravityValues` container.

::: gragra.analytic.water_waterfall_transfer_g_free

Compute G-free acceleration transfer functions for vertical cross-sectional area perturbations in a free-falling waterfall.

::: gragra.analytic.water_channel_uncorrelated_cell_coupling_g_free

Compute the geometric coupling coefficient for narrow-channel uncorrelated surface cells.
Classified as `INDEPENDENT_DERIVATION` (independent first-principles derivation in `gragra`);
external draft presentation notes are designated `ORIGINAL_SOURCE_UNVERIFIED`.

::: gragra.analytic.water_edge_step_transfer_g_free

Compute the G-free deterministic acceleration Fourier spectrum for a Heaviside step displacement at a channel inflow edge.

## Physical observables and spectral densities

High-level SI observables and ASD estimators from `gragra.observables`.

::: gragra.observables.water_channel_advection_observables

Compute physical SI gravity potential, acceleration, and gradient perturbations for channel surface displacement advection.

::: gragra.observables.water_channel_advection_dc_limit

Compute physical SI acceleration and gravity gradient perturbations at exact DC for water channel displacement.

::: gragra.observables.water_channel_advection_asd

Compute the amplitude spectral density (ASD) of gravitational acceleration induced by water channel surface fluctuations.

::: gragra.observables.water_channel_advection_strain_asd

Compute the gravitational-wave strain equivalent ASD for water channel advection.

::: gragra.observables.water_channel_kolmogorov_surface_asd

Evaluate Kolmogorov turbulent surface displacement ASD model for open water channels.

::: gragra.observables.water_waterfall_asd

Compute gravitational acceleration ASD for vertical waterfall cross-sectional area fluctuations.

::: gragra.observables.water_waterfall_strain_asd

Compute strain equivalent ASD for vertical waterfall Newtonian noise.

::: gragra.observables.water_channel_uncorrelated_asd

Compute acceleration ASD from spatially uncorrelated surface cells in a narrow water channel.

::: gragra.observables.water_edge_step_fourier_amplitude

Compute physical deterministic acceleration Fourier amplitude spectrum for an edge step displacement.

::: gragra.observables.water_edge_step_strain_fourier_amplitude

Compute deterministic strain Fourier amplitude spectrum for an edge step displacement.
