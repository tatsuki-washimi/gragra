
# What gragra computes

## The forward problem

gragra evaluates the Newtonian gravitational field produced by a mass
distribution you supply. Given source positions and weights (or a
displacement/density-perturbation field, or a discretized geometry), it
integrates the point-mass kernel over that distribution and returns the
potential, acceleration, projected acceleration, or gravity-gradient tensor
at the observation points you specify. It does not solve for the mass
distribution given a field — only the forward direction.

## The four abstractions

Every calculation in gragra is organized along four responsibilities:
**Source → Geometry → Integrator → Observable**.

### Source

A collection of mass elements: positions and weights (`WeightedPointSource`,
`PointMass`, `PointMassCloud`), or a converted continuous distribution
(volume element source, surface mass sheet). Weights are mass elements in
kilograms — signed or complex, in the general case — never a pre-integrated
physical quantity. See [Point sources](../guide/point-sources.md).

### Geometry

Shapes (`Sphere`, `Cuboid`, `Cylinder`, imported meshes) and their
discretization into point-like mass elements. Geometry answers "where is the
mass", not "how strong is the field there" — that is the integrator's job.
See [Geometry and discretization](../guide/geometry.md).

### Integrator

The layer that dispatches a normalized source and target to a kernel
contract and sums the geometric factor over sources: `Σⱼ wⱼ·K`. The
integrator and the kernels beneath it never multiply by the gravitational
constant — that is deliberate, so that the same integration machinery could
in principle be reused for a different force law.

### Observable

The public API. Multiplying the integrator's G-free result by `G_SI` is the
**only** thing an observable function does beyond dispatch — see
[Coordinates, signs and units](conventions.md) for `G_SI`'s value and
[Observables](../guide/observables.md) for the four observable quantities
themselves.

## Kinematic field constructors

gragra also ships closed-form generators for a handful of standard
*kinematic* fields — plane waves, Rayleigh eigenfunctions, stochastic P/S
realizations (see [Waves](../guide/waves.md)) — that you can use as input to
the Source/Geometry layers. Generating one of these fields is not the same
as solving the physics that would produce it: gragra does not solve an
elastodynamic, fluid, or detector-response PDE. If your field comes from
somewhere else — a solver, a measurement, a different simulation — you
supply it yourself (see [External field data and HDF5](../external/hdf5-fields.md)
and, for one specific solver, [SPECFEM3D Cartesian](../external/specfem3d.md))
and gragra integrates it.

## What is outside the model

- **Any dynamical PDE.** No elastodynamics, no fluid flow, no detector
  response. gragra consumes a displacement or density-perturbation field;
  it does not produce one from first-principles physics beyond the
  kinematic constructors above.
- **A general-purpose multi-solver platform.** SPECFEM3D Cartesian has an
  export-only native path, and the `mesh` extra has a bounded Foundation
  OpenFOAM 14 patch 20260724 legacy-ASCII `.vtk` density reader. Neither path
  supports native `polyMesh`, general OpenFOAM formats, or CFD-solver runs;
  see [External field data and HDF5](../external/hdf5-fields.md).
- **Inversion.** Given a field, gragra does not recover the source that
  produced it.
- **Time or frequency semantics.** gragra's core has no concept of a
  sampling rate, a spectral type, or an FFT sign convention — a batch axis
  on an imported field is an opaque index, not a time axis; see
  [Arrays, dtypes and axes](arrays.md).
