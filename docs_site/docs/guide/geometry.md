
# Geometry and discretization

## What the geometry layer provides

`gragra.geometry` describes shape and coordinates; density belongs to the
source construction step. The C5 high-level sphere workflow uses Quantity
inputs and an explicit `Frame` and sends the resulting point source to a
high-level evaluation model. See [What gragra computes](../concepts/model-and-scope.md)
and the [core API reference](../api/core.md).

## Primitive shapes

`gragra.geometry.Sphere` is a pure shape with a Quantity center, a Quantity
radius and a `Frame`. It has no density or grid-spacing fields. The current
high-level source factory supports uniform spheres:

```python
--8<-- "snippets/readme_geometry_primitives.py"
```

The older midpoint numerical primitive remains at `gragra.geometry.primitives`
for internal and lower-level workflows. Do not mix that numeric `Sphere` with
the high-level class of the same name.

## Discretization and cell-centre representation

The sphere discretizer currently uses **cell-centre inclusion** on a midpoint
Cartesian grid: a cell is included if its centre falls inside the shape, and
the representative mass element sits at that centre. It does not apply a
boundary-cell correction or renormalize total mass. The discrete mass
generally differs from the analytic mass at finite spacing; the measured
Sphere sequence and its limits are stated below.

## Discretization error

The recorded Sphere sweep covers one case only: radius `1 m`, density
`1000 kg/m³`, one exterior target at `(2.7, 0.3, -0.4) m`, and spacings
`R/[10, 20, 40, 80]`. Its finest errors pass a `0.05` relative-error gate for
each observable, and potential improves from coarsest to finest. The fitted
slope is reported but not asserted; errors need not decrease at every step.
These checks do not establish a convergence rule for other geometries,
targets, or source discretizations. See [analytic benchmarks](analytic-benchmarks.md).
The full numeric table and reproduction commands are kept in the internal
repository note `docs/validation/nonanalytic_static_sources.md`.

## Surface meshes

The sphere source above is volumetric. For a surface (a boundary, an
interface, an imported triangulated mesh), see
[Using external meshes](meshes.md) — surface meshes have their own
orientation and manifold-quality concerns that a volumetric primitive does
not.

## Status and verification

The geometry input has no top-level public Verification Block of its own.
The existing discretized-Sphere route and the newer raw, grid, and volume-mesh
`point_sum` routes have separate evidence and maturity; see the
[static-source Verification Blocks](static-sources.md#verification).
