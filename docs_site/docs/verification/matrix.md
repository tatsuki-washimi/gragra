
# Verification matrix

## How to read this table

Each row links to a full Verification Block on its owning page — this table
is an index, not a duplicate. "Agreement" is abbreviated; the linked block
carries the full tolerance, test node IDs, and known limits. See
[How to read a Verification Block](index.md#how-to-read-a-verification-block)
for what each field means and why a bare test count does not belong here.

## Matrix

| Symbol | Status | Agreement (summary) | Block |
|---|---|---|---|
| `G_SI` | Stable | n/a — CODATA literal | [Coordinates, signs and units](../concepts/conventions.md#status-g-si) |
| `WeightedPointSource` | Stable | exact | [Point sources](../guide/point-sources.md#status-weightedpointsource) |
| `PointMass` | Stable | exact at unit fixed point | [Point sources](../guide/point-sources.md#status-pointmass) |
| `PointMassCloud` | Stable | exact (sum of individual masses) | [Point sources](../guide/point-sources.md#status-pointmasscloud) |
| `point_sum` with raw static sources | Experimental | PointMass agrees with the exterior analytic Sphere result after ID/unit alignment (`<1e-12`); fixed point `rtol=1e-14` | [Static sources](../guide/static-sources.md#status-point-sum-raw-static) |
| `DensityPerturbationGrid` → `point_sum` | Experimental | two-cell signed discrete point oracle (`rtol=2e-14`) | [Static sources](../guide/static-sources.md#status-density-grid-static-point-sum) |
| `VolumeMesh` → `point_sum` | Experimental | unit tetra centroid-mass point oracle (`rtol=2e-14`) | [Static sources](../guide/static-sources.md#status-volume-mesh-static-point-sum) |
| `TargetPoints` | Stable | exact | [Point sources](../guide/point-sources.md#status-targetpoints) |
| `point_potential` | Stable | exact at unit fixed point | [Observables](../guide/observables.md#status-point-potential) |
| `point_acceleration` | Stable | exact at unit fixed point; `rtol=1e-15` chunk invariance | [Observables](../guide/observables.md#status-point-acceleration) |
| `point_gravity_gradient` | Stable | tensor-sign fixed point at `rtol=1e-14`; zero-softening trace within `32 ε max(abs(T))` for the tested configuration | [Observables](../guide/observables.md#status-point-gravity-gradient) |
| `coherent_point_acceleration` | Experimental | `rtol=1e-12` backend equivalence; exact at documented fixed points | [Coherent responses](../guide/coherent.md#status-coherent-point-acceleration) |
| `coherent_point_gravity_gradient` | Experimental | exact tensor-sign fixed point; `rtol=1e-15` chunk invariance | [Coherent responses](../guide/coherent.md#status-coherent-point-gravity-gradient) |
| `coherent_gravitoelastic_tensor` | Experimental | absolute-value golden at `G¹`; exact S-wave structural zero | [Coherent responses](../guide/coherent.md#status-coherent-gravitoelastic-tensor) |
| `coherent_acceleration_csd` | Experimental | absolute-value golden at `G²`; exact phase fixed point | [Coherent responses](../guide/coherent.md#status-coherent-acceleration-csd) |
| `displacement_transfer_acceleration` | Experimental | exact sign fixed point; bulk+surface equivalence converges with distance | [Displacement and density-perturbation fields](../guide/fields.md#status-displacement-transfer-acceleration) |
| `displacement_transfer_potential` | Experimental | symbolic derivative, fixed-point, and NumPy/Numba checks; `rtol=1e-12` | [Displacement and density-perturbation fields](../guide/fields.md#status-displacement-transfer-derivatives) |
| `displacement_transfer_gravity_gradient` | Experimental | symbolic derivative, symmetry, softened-trace, and NumPy/Numba checks | [Displacement and density-perturbation fields](../guide/fields.md#status-displacement-transfer-derivatives) |
| `displacement_field_acceleration` | Experimental | independent linearized and central-difference checks on synthetic GLL data | [Displacement and density-perturbation fields](../guide/fields.md#status-displacement-field-acceleration) |
| finite P1 tetra body | Experimental | mass conservation, topology rejection, G-free representation comparison, refinement, and Taylor checks | [Displacement and density-perturbation fields](../guide/fields.md#finite-piecewise-tetra-bodies) |
| `trajectory_potential` | Experimental | static trajectory matches the point-source result | [Prescribed motion verification](../api/core.md#status-prescribed-motion) |
| `trajectory_acceleration` | Experimental | static trajectory and NumPy/Numba checks | [Prescribed motion verification](../api/core.md#status-prescribed-motion) |
| `trajectory_gravity_gradient` | Experimental | static trajectory and collision checks | [Prescribed motion verification](../api/core.md#status-prescribed-motion) |
| `rigid_transform_potential` | Experimental | rigid translation and rotation covariance | [Prescribed motion verification](../api/core.md#status-prescribed-motion) |
| `rigid_transform_acceleration` | Experimental | rigid translation and rotation covariance | [Prescribed motion verification](../api/core.md#status-prescribed-motion) |
| `rigid_transform_gravity_gradient` | Experimental | rigid translation and rotation covariance | [Prescribed motion verification](../api/core.md#status-prescribed-motion) |
| `backend="numpy"` | Stable | reference implementation; exact at fixed points | [Compute backends](../guide/backends.md#status-backend-numpy) |
| `backend="numba"` | Experimental | `rtol=1e-12, atol=1e-15` vs. NumPy | [Compute backends](../guide/backends.md#status-backend-numba) |
| `backend="cpp"` | Experimental | `rtol=1e-12, atol=1e-15` vs. NumPy, three-way with numba | [Compute backends](../guide/backends.md#status-backend-cpp) |
| `backend="cupy"` | Experimental | `rtol=1e-12, atol=1e-15` vs. NumPy — local GPU only, not run in CI | [Compute backends](../guide/backends.md#status-backend-cupy) |
| `gragra.adapters` HDF5 field I/O | Experimental | exact round trip on synthetic fixtures | [External field data and HDF5](../external/hdf5-fields.md#status-adapters-hdf5-field-io) |
| `gragra.adapters.specfem` | Experimental | pilot A1–A3 pass; new 12/18 km acceleration arithmetic complete, nested gate rejected; A4 continuity does not pass | [SPECFEM3D Cartesian](../external/specfem3d.md#status-adapters-specfem) |
| `gragra.adapters` GWexpy observable bridge | Experimental | 30/30 round trips for GWexpy 0.1.7 and 0.1.14 | [GWexpy observable bridge verification](../api/experimental.md#status-gwexpy-observable-bridge) |

## Symbols with no verification claim

The rows above index the current symbols with an 8-field Verification Block.
That schema is reserved for a deliberately small set of symbols users are
most likely to call first. Other public symbols are documented through their
own docstrings on the [API reference](../api/physics.md) and
[Additional APIs and internal modules](../api/experimental.md), without a
duplicate admonition here. Their test evidence and limits are specific to
those APIs; this table does not make a suite-wide coverage claim. See
[Analytic benchmarks](../guide/analytic-benchmarks.md#status-and-verification)
and [Waves](../guide/waves.md#status-and-verification) for the two pages
that state this distinction explicitly for their own symbols.
