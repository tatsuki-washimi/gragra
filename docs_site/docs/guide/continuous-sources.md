
# Volume and surface sources

## From a continuous distribution to mass elements

A continuous density perturbation or surface mass sheet is not a point
source until it is discretized into mass elements. `VolumeElementSource` and
`SurfaceMassSheetSource` (both in `gragra.gravity_sources`) hold the discretized
representation and convert to a `WeightedPointSource` via
`as_weighted_source()` — the same generic source type used everywhere else
(see [Point sources](point-sources.md)).

## Volume sources

`VolumeElementSource` represents a set of discretized volume elements: a
centroid, a cell volume, and a density perturbation per element. Its
`weights_kg` (via `as_weighted_source()`) are mass elements only —
`δm = δρ·dV` — with the same "mass element, not a transfer coefficient"
discipline described in [Point sources](point-sources.md#what-the-source-weights-mean).

## Surface sources

`SurfaceMassSheetSource` represents displacement-induced surface mass
perturbations: a mass sheet living on a surface, built from a face normal,
a face area, and a density jump or displacement value at that face. The
sign of the surface normal is significant — see
[Using external meshes](meshes.md) for how orientation is checked and
corrected on an imported mesh.

## Combining a bulk and a surface contribution

`δρ = −∇·(ρ₀ξ)` decomposes into a bulk term (the divergence of the
displacement field weighted by background density) and a surface/interface
term (the jump in displacement across a density discontinuity). Both
contributions must be computed and added when both are physically present —
using only one silently gives an incomplete, not merely approximate,
answer. The six equivalence conditions that must hold for a bulk/surface
pair to agree with a closed manifold's exact field (closed manifold,
consistent winding, verified outward normals, sign convention, bulk/surface
geometric agreement, background-density truncation) are listed in
[Displacement and density-perturbation fields](fields.md), along with the
important special case where the bulk term is *exactly* zero for a rigid
translation.

## Convergence with refinement

A discretized volume or surface source is an approximation to the exact
continuous integral. Refining the discretization (more, smaller elements)
should converge the computed field toward the closed-form or reference
value — see [Analytic benchmarks](analytic-benchmarks.md) for the
convergence-sweep methodology and what a non-converging refinement
indicates (a bug, not expected discretization noise).

## Status and verification

Neither `VolumeElementSource` nor `SurfaceMassSheetSource` owns a top-level
public symbol with its own Verification Block — both convert into
`WeightedPointSource`, whose block is on
[Point sources](point-sources.md#status-weightedpointsource). Discretization
accuracy for the geometry primitives that commonly feed these sources
(`Sphere`, `Cuboid`, `Cylinder`) is covered on
[Geometry and discretization](geometry.md) and
[Analytic benchmarks](analytic-benchmarks.md).
