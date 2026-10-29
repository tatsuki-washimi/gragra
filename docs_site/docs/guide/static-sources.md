# Static point, grid and mesh sources

`point_sum` connects existing static source representations to the high-level
`.at(targets).run()` workflow. It accepts a discretized `Sphere`, caller-made
real point sources, and `VolumeElementSource` values produced from density
grids or volume meshes. Existing imports and low-level observable calls remain
available; the high-level path adds `EvaluationResult`, target IDs, units,
Frames, typed field meaning, and provenance.

## Raw static point sources

Bare inputs such as `PointMass`, `PointMassCloud`, `WeightedPointSource`, and
`VolumeElementSource` do not identify a coordinate Frame. This minimal example
uses the `PointMass` convenience input, which has a default source-field
meaning; weighted and volume sources also need an explicit field definition.

```python
from astropy import units as u

from gragra.geometry import Frame
from gragra.gravity_sources import PointMass, point_sum
from gragra.targets import TargetPoints

frame = Frame("local")
source = PointMass(2.0, [0.0, 0.0, 0.0])
targets = TargetPoints([[3.0, 4.0, 0.0]] * u.m, ("far",), frame)
result = point_sum(source, frame=frame).at(targets).run()
```

`PointMass` and `PointMassCloud` default to a total source field whose
potential is zero at infinity. A general weighted or volume source also needs
an explicit `GravityFieldDefinition`; this states whether the signed weights
represent a source field or a field change. The static route accepts only real,
finite mass weights. Complex weights do not by themselves mean a phasor.

`provenance` is an immutable record of caller-supplied context. Values are
`None`, strings, booleans, integers, finite real scalars, or flat tuples of
finite real numbers. A three-value `origin_m` is accepted as provenance, but
does not translate coordinates. Likewise, a string such as
`coordinate_frame="local"` does not connect two Frames or rotate coordinates.
Supply a `Frame` whose origin and orientation describe the positions, and use
connected Frames for targets.

## Density grids and volume meshes

The static density-grid route is:

```text
DensityPerturbationGrid
  -> VolumeElementSource
  -> point_sum(source, frame=..., field_definition=..., provenance=...)
  -> EvaluationResult
```

For a mesh, `to_volume_element_source` obtains cell centroids and volumes,
then the source is bound through the same static path:

```text
VolumeMesh
  -> cell centroids and volumes
  -> VolumeElementSource
  -> WeightedPointSource mass elements
  -> point_sum(...)
```

These routes evaluate the resulting point-mass representation at each target.
The input mesh or grid is not integrated as a continuous body by `point_sum`.
The external coordinate labels, mesh identifiers, and other import facts belong
in provenance; the caller-provided `Frame` controls the coordinate transform.

The executable walkthrough covers both conversions using an in-memory
`VolumeMesh`, so it does not require the `mesh` extra. File-based mesh readers
from `meshio` require that extra.

```python
--8<-- "static_sources_workflow.py"
```

## Selecting and comparing results

`EvaluationResult.select(...)` makes the observable set explicit before
comparison. `compare(reference=..., candidate=...)` aligns targets by exact ID
sets, reorders candidate values by ID, converts compatible units, and checks
that target positions agree in their connected root Frame within its stated SI
tolerance. It requires compatible typed field meaning, representation,
interaction, G convention, and output component axes.

Comparison does not rotate components, interpolate target coordinates, or
infer a Frame from metadata. Source count, Sphere spacing, mesh identity, and
execution-record backend/version fields are not physical compatibility keys;
they remain available as diagnostics or provenance. The result of comparison
shows differences between the supplied discrete representations. It does not
establish their accuracy relative to a continuous body.

## Verification

### `point_sum` with raw static sources { #status-point-sum-raw-static }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Experimental |
    | **Verified against** | closed-form analytic solution; invariance property |
    | **Agreement** | A PointMass result matches the exterior analytic Sphere result at the sampled targets with relative difference below `1e-12` after ID and unit alignment; the single-point closed form is separately checked at `rtol=1e-14`. |
    | **Test evidence** | `tests/high_level/test_static_source_compare.py::test_point_sum_compares_with_exterior_analytic_sphere_after_id_alignment`; `tests/high_level/test_point_sum_sources.py::test_point_mass_run_matches_independent_point_oracle_and_context` |
    | **Conditions** | Real static point masses, target positions outside the unit Sphere, shared connected Frame root, potential zero at infinity, and NumPy execution. |
    | **Not verified** | Accuracy of a generic point cloud as an approximation to a continuous body, near-source accuracy, complex or time-varying weights, and parity with other compute backends. |
    | **Known limits** | Bare inputs require an explicit Frame. Weighted and volume sources require a `GravityFieldDefinition`; complex weights are rejected; static point kernels require potential reference `ZERO_AT_INFINITY`; the raw point model has no `plot_setup()` method. `provenance` does not transform coordinates. |
    | **Scope** | The raw static input route is Advanced / Experimental. The existing discretized-Sphere route retains its separate evidence and maturity. |

### `DensityPerturbationGrid` to static point result { #status-density-grid-static-point-sum }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Experimental |
    | **Verified against** | closed-form analytic solution; invariance property |
    | **Agreement** | The two-cell signed grid fixed point agrees with its analytic point-mass sum to `rtol=2e-14` for potential, acceleration, and gradient. |
    | **Test evidence** | `tests/high_level/test_static_density_mesh.py::test_density_grid_static_result_signed_oracle_provenance`; `tests/high_level/test_static_source_compare.py::test_density_grid_and_mesh_results_compare_with_distinct_provenance` |
    | **Conditions** | Two constant-density cells, one connected source/target coordinate tree with an explicit source Frame, finite targets outside the cells, and NumPy execution. |
    | **Not verified** | Continuous-cell integration error, arbitrary grid resolution, near-interface accuracy, coordinate transformation inferred from adapter metadata, and other backends. |
    | **Known limits** | Grid cells become centroid point masses. A field-change definition and the physical coordinate Frame are caller inputs; provenance strings do not establish either one. |
    | **Scope** | Experimental static density-grid workflow; time series and phasors are outside this route. |

### `VolumeMesh` to static point result { #status-volume-mesh-static-point-sum }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Experimental |
    | **Verified against** | closed-form analytic solution; invariance property |
    | **Agreement** | A unit tetrahedron at density `6 kg/m³` produces a centroid mass of `1 kg`; its discrete point field agrees with the analytic point-mass formula to `rtol=2e-14` for the tested exterior target. |
    | **Test evidence** | `tests/high_level/test_static_density_mesh.py::test_volume_mesh_static_result_tetra_oracle_provenance`; `tests/high_level/test_static_source_compare.py::test_density_grid_and_mesh_results_compare_with_distinct_provenance` |
    | **Conditions** | One unit tetrahedron, density `6 kg/m³`, explicit source Frame, exterior target, and NumPy execution. |
    | **Not verified** | Continuous tetrahedron field accuracy, mesh refinement, near-interface targets, arbitrary imported mesh formats, and other backends. |
    | **Known limits** | The route uses cell centroids and volumes to form point masses; it does not add finite-body or near-interface guarantees. Import metadata remains provenance. |
    | **Scope** | Experimental static `VolumeMesh` conversion and evaluation workflow. |
