# gragra

## What gragra computes

gragra computes Newtonian gravity perturbations by forward integration. You supply a mass distribution, a mass-density perturbation, or a displacement field; gragra returns the gravitational potential, acceleration, and gravity-gradient tensor at the observation points you specify. Its most common use is estimating seismic Newtonian-noise contributions to gravitational-wave detectors.

Every calculation follows the same four steps: a **source** (points, volumes, surfaces, or a field you loaded), a **geometry** that discretizes it, an **integrator** that sums the geometric kernel against the source weights, and an **observable** that turns that sum into a physical quantity in SI units. What the layers are and why the boundary between them is drawn where it is are described in [What gragra computes](concepts/model-and-scope.md); the sign, unit and axis conventions that every result depends on are in [Coordinates, signs and units](concepts/conventions.md).

## What gragra does not do

gragra does not solve a dynamical PDE — not elastodynamics, not fluid flow, not detector response. It ships closed-form *kinematic* field constructors (plane waves, Rayleigh eigenfunctions, stochastic P/S realizations) that you can use as input, but for anything else you supply the field yourself and gragra integrates it.

gragra is not a multi-solver platform. SPECFEM3D Cartesian has a native
export path and documented narrow HDF5 reader. The `mesh` extra also provides
a bounded Foundation OpenFOAM 14 patch 20260724 legacy-ASCII `.vtk` density
adapter. Neither path supports general solver formats, native `polyMesh`, or
CFD-solver runs; see [Limitations and non-guarantees](verification/limitations.md).

Both statements have consequences that are easy to discover too late — which physical cases have a dedicated analytic treatment and which do not, which code paths are checked automatically and which are checked only on one machine. They are collected in [Limitations and non-guarantees](verification/limitations.md), which is the page to read before you rely on a number.

## Research scope

本リポジトリは個人の科学研究用であり、数値実装・検証・研究計算を目的とする。
公開・配布準備は対象外とする。将来必要になった場合も、その時点のコードから
独立に検討し、過去の公開手続き・認定記録との整合は要求しない。

[Installation](getting-started/installation.md) covers the source install and
the optional extras.
Record the exact commit you installed because a development version alone does
not identify the contents of a changing research tree.

## Where to start

Three routes through the documentation, chosen by what you already have in hand. Each is two pages long; everything else can wait.

### I have a seismic displacement field

Your field comes from an external solver, a measurement, or your own code, and you want the gravity perturbation it produces.

1. [Displacement and density-perturbation fields](guide/fields.md) — how a displacement field becomes a mass-density perturbation, why the bulk and surface contributions are separate, and what happens to each when your input is incomplete.
2. [External field data and HDF5](external/hdf5-fields.md) — the file layouts gragra reads, and what you have to convert yourself.

If your field comes from SPECFEM3D Cartesian, read [SPECFEM3D Cartesian](external/specfem3d.md) as well, and read it before you plan the work: it states exactly where the supported path ends.

### I have point masses or simple shapes

You are modelling vehicles, equipment, water bodies, or any object whose motion you can describe directly.

1. [Minimal example](getting-started/minimal-example.md) — a complete calculation in a few lines, with the result checked against the closed-form value.
2. [Geometry and discretization](guide/geometry.md) — turning a shape into source points, and how the discretization error behaves.

From there, [Point sources](guide/point-sources.md) and [Observables](guide/observables.md) document the source types and the four point observables in full.

### I want plane waves and coherent statistics

You are working in the frequency domain with complex amplitudes, directional averages, or cross-spectral quantities.

1. [Waves](guide/waves.md) — plane waves, polarizations, Rayleigh eigenfunctions, and the complex-amplitude convention they all share.
2. [Coherent responses and cross statistics](guide/coherent.md) — coherent observables, the gravitoelastic tensor, and cross-spectral densities, including which of them carry which power of the gravitational constant.

[Complex outputs](concepts/complex-outputs.md) explains what a complex result means here, and what it does not mean.

## How to read the verification claims on this site

Every page that documents a public symbol carries a **Verification Block** for it: a fixed eight-field table naming what the result was checked against, at what level of agreement, by which test, under which conditions — and, in its own field, what that evidence does *not* cover. The vocabulary is closed and the test references are machine-checked, so a block cannot quietly become a paragraph of reassurance.

Read [How gragra is verified](verification/index.md) first. It explains why the size of the test suite is not the evidence, and gives, method by method, the kind of error each check catches and the kind it is structurally blind to. [Verification matrix](verification/matrix.md) is the index of every block; [Literature coverage](verification/coverage.md) says how much of the standard reference catalogue of physical cases has a dedicated treatment here; [Limitations and non-guarantees](verification/limitations.md) collects what none of it covers.
