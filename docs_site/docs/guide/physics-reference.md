# Physics reference

This page is a map to the physical models documented in `gragra`. It does not
replace their equations, assumptions, variable definitions or source notes;
those stay with the model-specific guides linked below. The
[models and methods catalogue](model-and-method-catalogue.md) is the compact
index of model families, geometries, observables and related API paths.

## Shared conventions

Before comparing formulas, establish what the source and observable mean in
the calculation:

- [What gragra computes](../concepts/model-and-scope.md) defines the forward
  problem and the boundary between supplied fields and gravity evaluation.
- [Coordinates, signs and units](../concepts/conventions.md) sets the
  coordinate, sign and SI conventions.
- [Observables](observables.md) defines potential, acceleration and the
  gravity-gradient tensor, including where the gravitational constant enters.
- [Complex outputs](../concepts/complex-outputs.md) explains phasor values and
  their time and phase references.

## Find a model by physical problem

| Physical problem | Model-specific material | Read there for |
|---|---|---|
| Gravity from finite, uniform-density bodies | [Analytic benchmarks](analytic-benchmarks.md); [Geometry and discretization](geometry.md) | Closed forms for the documented sphere, cuboid and cylinder cases; boundary conditions; the separate point-discretization route |
| Gravity from a supplied displacement or density-perturbation field | [Displacement and density-perturbation fields](fields.md); [Volume and surface sources](continuous-sources.md) | Bulk and surface contributions, density conventions, and the conditions for comparing the two forms |
| Seismic body waves, Rayleigh waves and cavities | [Waves](waves.md); [Gen1 analytic models](gen1-analytic-models.md); [Analytic benchmarks](analytic-benchmarks.md) | Field conventions, model-specific equations and assumptions, and the cases with a documented analytic reference |
| Atmospheric sound and temperature perturbations | [Gen1 analytic models](gen1-analytic-models.md); [Stochastic Newtonian noise](stochastic-nn.md) | Prescribed deterministic fields versus stochastic spectra, with their input conditions |
| Water-flow Newtonian noise | [Water Newtonian noise](water-newtonian-noise.md) | Channel and waterfall geometries, model inputs and spectral assumptions |
| Coherent responses and cross statistics | [Coherent responses and cross statistics](coherent.md); [Stochastic Newtonian noise](stochastic-nn.md) | Phasor conventions, directional averages, and the powers of $G$ in first- and second-order quantities |

The linked guides state the variables and applicability conditions for their
own models. A model's presence in the catalogue is an inventory entry; it does
not establish scientific acceptance. See [How gragra is verified](../verification/index.md),
[Literature coverage](../verification/coverage.md), and
[Limitations and non-guarantees](../verification/limitations.md) for the
scope of the evidence already recorded.

## Scope of the physical model

`gragra` evaluates gravity from source distributions, geometries and fields
that you supply. It does not solve an elastodynamic, fluid-flow or detector
response equation. Its kinematic wave constructors are inputs to the gravity
calculation, not solutions produced by a general-purpose PDE solver. See
[What gragra computes](../concepts/model-and-scope.md) before applying a model
to a new physical configuration.

This reference only organizes existing public documentation. It adds no new
model or verification result.
