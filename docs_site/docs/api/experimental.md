# Additional APIs and internal modules

## Interface status

The adapters, field I/O helpers, and optional integrations are
documented here. "Experimental" describes an interface that may change as
research use exposes new requirements; it does not mean untested. The
SPECFEM3D section describes the supported input profile and its scientific
limitations. [Compute backends](../guide/backends.md) and
[Verification](../verification/index.md) describe the numerical checks;
[External data and HDF5](../external/hdf5-fields.md) and
[SPECFEM3D Cartesian](../external/specfem3d.md) state each adapter's actual
scope in full, and take precedence over any summary here.

## Adapters

### Meshes (`meshio`/`trimesh`, extras `mesh` / `mesh-quality`)

::: gragra.adapters.TriangleSurfaceMesh

A triangle surface mesh — vertices, triangle connectivity, and (once
computed) per-facet centroids, normals and areas — the container
`from_meshio` and `gragra.geometry.PrimitiveSurfaceMesh` share a shape with.

::: gragra.adapters.VolumeMesh

A volume mesh — vertices, cell connectivity, and (once computed) per-cell
centroids and volumes — the volume counterpart to `TriangleSurfaceMesh`.

::: gragra.adapters.from_meshio

Reads a mesh file via `meshio` and extracts its triangle cells into a
`TriangleSurfaceMesh`, requiring the `mesh` extra.

::: gragra.adapters.from_meshio_volume

Reads a mesh file via `meshio` and extracts its volume cells into a
`VolumeMesh`.

::: gragra.adapters.centroids_normals_areas

Computes per-triangle centroids, unit normals and areas for a
`TriangleSurfaceMesh` — the geometric data a surface source needs.

::: gragra.adapters.cell_centroids_volumes

Computes per-cell centroids and volumes for a `VolumeMesh` — the geometric
data a volume source needs.

::: gragra.adapters.select_group

Filters mesh cells belonging to a named group (as recorded by the source
mesh format), for meshes with multiple named regions.

::: gragra.adapters.select_group_tag

Filters mesh cells by an integer group tag, the numeric-tag counterpart to
`select_group`.

::: gragra.adapters.to_surface_mass_sheet

Converts a `TriangleSurfaceMesh` into a `SurfaceMassSheetSource`, the step
that hands mesh geometry to the source layer described on
[Sources, targets and geometry](core.md).

::: gragra.adapters.to_volume_element_source

Converts a `VolumeMesh` into a `VolumeElementSource`, the volume-mesh
counterpart to `to_surface_mass_sheet`.

::: gragra.adapters.to_unstructured_displacement_field

Builds an `UnstructuredDisplacementField` from the tetra blocks of a
`VolumeMesh`, preserving their input order and rejecting nonempty hex blocks.

::: gragra.adapters.cell_density_from_group_tags

Maps every tetra cell-group tag to one finite, non-negative density while
requiring the supplied mapping to cover exactly the tags that the mesh uses.

### OpenFOAM density snapshots (`mesh` extra)

The supported profile is Foundation OpenFOAM 14 patch 20260724, reconstructed
single-region legacy ASCII `UNSTRUCTURED_GRID` VTK. It accepts native cell data
`C`, `Vc`, `cellID`, and a caller-selected density field. A file may contain
both `rho` and `delta_rho`; `field_name` selects exactly one of them.

::: gragra.adapters.OpenFOAMDensitySnapshot

An immutable loader-created snapshot retaining native cell centres, volumes,
IDs, units, time assertion, frame origin, and frozen producer provenance.

::: gragra.adapters.load_openfoam_density

Loads one bounded VTK snapshot after validating caller-supplied units and
producer profile assertions. World coordinates equal case coordinates plus the
declared origin; no CFD solver is run and pressure or temperature is not read.

::: gragra.adapters.check_mesh_quality

Computes geometric quality metrics for a surface mesh (degenerate-element
and aspect-ratio checks), requiring the `mesh-quality` extra.

::: gragra.adapters.load_trimesh

Loads a `trimesh.Trimesh` from a file path or a meshio-like mesh object,
requiring the `mesh-quality` extra.

### Directional quadrature (`scipy`, extra `lebedev`)

::: gragra.adapters.lebedev_grid

Generates a Lebedev spherical-quadrature grid (directions and normalized
weights) for the directional averages in
[guide/coherent.md](../guide/coherent.md), requiring the `lebedev` extra
(`scipy>=1.15.0`, where `lebedev_rule` was added).

### External field data (`h5py`, extra `field-io`)

External field HDF5 I/O is documented narratively in
[External field data and HDF5](../external/hdf5-fields.md); this is the
symbol-level reference.

::: gragra.adapters.DiscreteBackgroundDensity

A background density model (implementing the `BackgroundDensityModel`
protocol from [gragra.fields](physics.md#fields)) backed by discrete,
imported density samples rather than an analytic function.

::: gragra.adapters.DifferentiableDiscreteBackgroundDensity

The differentiable variant of `DiscreteBackgroundDensity`, additionally
supplying ∇ρ₀ from the imported samples.

::: gragra.adapters.from_hdf5_field

Imports a displacement field — unstructured or structured, dispatched on
the file's recorded layout — from an HDF5 file conforming to gragra's
external field schema.

::: gragra.adapters.from_hdf5_displacement_grid

Imports a `gragra.fields.DisplacementGrid` from an HDF5 file, the
structured-grid-specific form of `from_hdf5_field`.

::: gragra.adapters.from_hdf5_unstructured_field

Imports a `gragra.fields.UnstructuredDisplacementField` from an HDF5 file,
the unstructured-mesh-specific form of `from_hdf5_field`.

::: gragra.adapters.read_hdf5_background_density

Reads background density data from an HDF5 file, for pairing with an
imported displacement field via a `BackgroundDensityModel`.

::: gragra.adapters.iter_hdf5_field_snapshots

Iterates over batched HDF5 field snapshots one at a time, for files using
the schema's leading batch axis (see
[External field data and HDF5](../external/hdf5-fields.md) for why on-disk
batch layout differs from the in-memory API shape convention).

::: gragra.adapters.from_hdf5_batched_weights

Computes batched mass-element weights directly from a batched HDF5 file,
combining `iter_hdf5_field_snapshots` with the weight-computation helpers on
[Fields](physics.md#fields) in one call.

### SPECFEM3D Cartesian GLL path (extra `field-io`)

Scope, supported input profile, and known limitations are the subject of
[SPECFEM3D Cartesian](../external/specfem3d.md) — read that
page before using either symbol below; this entry states only what each one
does mechanically.

::: gragra.adapters.ExportConfig

Configuration for one SPECFEM3D Cartesian GLL export: the supported profile or
time-sampled exploratory mode and the output path to use.

::: gragra.adapters.export_specfem_gll_hdf5

Exports one `mesh_type="gll"` HDF5 file from a SPECFEM3D Cartesian run's
native binary output, for the single profile `ExportConfig` selects.

::: gragra.adapters.read_specfem_gll_hdf5

Loads the rc1 time-sampled schema-v1 profile into a closed immutable GLL
dataset. PML, mixed material, and non-elastic input are rejected explicitly.

::: gragra.adapters.SpecfemGLLDataset

Stores the immutable element-ordered GLL coordinates, source masses,
displacement snapshots, time values, and source-file identity used by the
acceleration observable.

### GWexpy field bridge (base dependency, experimental)

`gragra`'s component-axis convention is compatible with GWexpy's
`VectorField`/`TensorField` axis convention, not shape-identical (GWexpy
fields are 4D) — see the caveat on [README.md](https://github.com/tatsuki-washimi/gragra/blob/main/README.md).
The numerical layer does not import GWexpy. GWexpy is a required runtime
dependency, while the adapters import it only when conversion is called.
The legacy converters, observable bridge, single-density-snapshot bridge,
and fixed-mesh series bridges below are the public GWexpy surface.

`to_gwexpy_observable` and `from_gwexpy_observable` preserve a current native
Field value with companion metadata for static, time-first, and
frequency-target-first multi-target observables.

The fixed-mesh bridges below exchange density, displacement, and surface
density series. They keep scalar weights distinct from mass-plus-vector
displacement, store geometry and IDs in the immutable sidecar, and read the
current native field values on reconstruction.

::: gragra.adapters.to_gwexpy_density_series

Converts a fixed-volume density time series or phasor series to a native
GWexpy scalar field with its complete immutable geometry and phase sidecar.

::: gragra.adapters.from_gwexpy_density_series

Reconstructs either fixed-volume density series type after validating units,
axis metadata, and the current native scalar field values.

::: gragra.adapters.to_gwexpy_density_time_series

Converts only a real fixed-volume density time series, rejecting phasors at
the typed entrypoint before native GWexpy conversion.

::: gragra.adapters.from_gwexpy_density_time_series

Restores only a real fixed-volume density time series and rejects a phasor
bundle even when its scalar layout is otherwise valid.

::: gragra.adapters.to_gwexpy_density_phasors

Converts only complex fixed-volume density phasors while retaining their
frequency, epoch, and spatial phase-origin metadata.

::: gragra.adapters.from_gwexpy_density_phasors

Restores only complex fixed-volume density phasors after checking their
frequency-specific phase metadata and current native scalar values.

::: gragra.adapters.to_gwexpy_displacement_series

Converts a fixed-mesh displacement series to a native GWexpy vector field,
with background mass remaining separately stored in the immutable sidecar.

::: gragra.adapters.from_gwexpy_displacement_series

Restores a fixed-mesh displacement series from current native vector
components while preserving component order, IDs, geometry, and axis data.

::: gragra.adapters.to_gwexpy_surface_density_series

Converts a fixed-quadrature surface-density series to a native scalar field,
retaining areas, normals, and face metadata outside the scalar values.

::: gragra.adapters.from_gwexpy_surface_density_series

Restores a fixed-quadrature surface-density series after validating native
units and its immutable surface geometry sidecar.

For frequency data, `exp(-iwt)` means
`u(t) = Re[C(f) exp(-i 2π f (t - t_ref_s))]`; `exp(+iwt)` uses the corresponding
plus sign. The bridge retains `t_ref_s` and `phase_origin_m` in companion
metadata and infers neither conjugation nor an epoch conversion.
`phase_origin_m` may be `None` only when no spatial phase origin applies.

::: gragra.adapters.GwexpyObservableBundle

Pairs a native GWexpy Field with recursively immutable companion metadata for
a gragra observable conversion.

::: gragra.adapters.ObservableArrayBundle

Pairs a read-only NumPy observable array with the immutable companion metadata
reconstructed from a GWexpy observable conversion.

::: gragra.adapters.to_gwexpy_observable

Converts a static, time-first, or frequency-target-first gragra observable
array and its positions, axis values, and metadata into a GWexpy Field bundle.

::: gragra.adapters.from_gwexpy_observable

Reads the current values from a GWexpy observable bundle and returns a
read-only gragra array bundle after validating its companion metadata.

### Single density snapshot bridge

This bridge accepts one real `rho` or `delta_rho` `VolumeElementSource` at one
time. It keeps irregular positions, volumes, IDs, frame, origin, and metadata
in an immutable sidecar; it is not a file reader or a time-series interface.

::: gragra.adapters.GwexpyDensitySnapshotBundle

The native scalar field plus sidecar for one density snapshot. A bare native
field is deliberately insufficient for reconstruction because it has no
irregular volume-element geometry.

::: gragra.adapters.DensitySnapshotArrayBundle

The reconstructed volume-element source and validated sidecar. Its `time_s`
property reads the single snapshot time from that sidecar.

::: gragra.adapters.to_gwexpy_density_snapshot

Creates the native `(1, N, 1, 1)` scalar density field and frozen sidecar from
one real density snapshot; conversion imports GWexpy only when called.

::: gragra.adapters.from_gwexpy_density_snapshot

Validates the field's current values, axes, units, and sidecar before
reconstructing a read-only `VolumeElementSource`; it performs no file I/O.

Earlier bridge measurements against GWexpy 0.1.7 and 0.1.14 are historical.
The current dependency range is `gwexpy>=0.2.3,<0.3`. Focused bridge,
import, and policy checks passed locally with GWexpy 0.2.3; this does not
establish scientific equivalence for every supported version or workflow.

### GWexpy observable bridge verification { #status-gwexpy-observable-bridge }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Experimental |
    | **Verified against** | invariance property |
    | **Agreement** | All three observables and accepted layouts round trip in the focused GWexpy 0.2.3 check. |
    | **Test evidence** | `tests/test_gwexpy_observable.py::test_round_trip_all_observables_and_layouts`, `tests/test_gwexpy_observable.py::test_inverse_reads_current_native_component_values`, `tests/test_gwexpy_observable_missing.py::test_module_import_is_lazy_and_conversion_reports_missing_dependency` |
    | **Conditions** | Focused local checks with GWexpy 0.2.3; adapter import remains lazy |
    | **Not verified** | Every version in the supported range and all scientific workflows |
    | **Known limits** | The bridge rejects implicit unit conversion and unsupported metadata values |
    | **Scope** | Static, time-first, and frequency-target-first multi-target observable bundles |

::: gragra.adapters.to_gwexpy_scalar_field

Converts gragra scalar output (e.g. `point_potential`) into a GWexpy
`ScalarField`.

::: gragra.adapters.to_gwexpy_vector_field

Converts gragra vector output (e.g. `point_acceleration`) into a GWexpy
`VectorField`.

::: gragra.adapters.to_gwexpy_tensor_field

Converts gragra tensor output (e.g. `point_gravity_gradient`) into a GWexpy
`TensorField`.

## Internal modules that are not part of the public contract

Anything under `gragra.*` not reachable from a package's `__all__` — private
helpers prefixed `_`, internal-only test fixtures, and backend-dispatch
plumbing such as `gragra.kernels._numba_*` / `gragra.kernels._cpp_*` /
`gragra.kernels._cupy_*` — is not part of the public API and can change or
disappear without a deprecation notice. If you find yourself importing one
of these directly, the public entry point you actually want is almost
always one layer up: [Top-level API](top-level.md),
[Sources, targets and geometry](core.md), or the Adapters section above.
