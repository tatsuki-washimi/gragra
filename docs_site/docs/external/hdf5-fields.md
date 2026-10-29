
# External field data and HDF5

`gragra.adapters` reads displacement and background-density fields from a
gragra-defined HDF5 layout. This is gragra's own file format for exchanging
fields with the outside world — it is not a reader for any particular
solver's native output. The [SPECFEM3D Cartesian](specfem3d.md) exporter is
the one bundled writer for this layout.

## Supported layouts

The file's root `mesh_type` attribute selects one of two layouts:

- **`"structured"`** — a regular or rectilinear grid, read by
  `from_hdf5_displacement_grid` into a `DisplacementGrid`. Required
  datasets: `x_m`, `y_m`, `z_m` (1-D coordinate axes) and `displacement_m`.
- **`"unstructured"`** — a tetrahedral mesh, read by
  `from_hdf5_unstructured_field` into an `UnstructuredDisplacementField`.
  Required datasets: `vertices_m`, `connectivity`, `displacement_m`.

`from_hdf5_field` reads the `mesh_type` attribute and dispatches to whichever
of the two applies, returning the matching field type.

There is no `mesh_type="surface"` layout yet — a single-snapshot or batched
surface HDF5 schema is not implemented. `to_surface_mass_sheet` (see
[Volume and surface sources](../guide/continuous-sources.md)) works from
in-memory mesh data, not from a file read by this module.

## Loading a field

This example requires `gragra[field-io]` (`h5py`) and is **not executed by
the test suite** — see [examples/09_external_field_density.py](https://github.com/tatsuki-washimi/gragra/blob/main/examples/09_external_field_density.py)
for the runnable, tested version.

```python title="requires gragra[field-io]; not executed by the test suite"
from gragra.adapters import from_hdf5_field, read_hdf5_background_density

field = from_hdf5_field("field.h5")               # DisplacementGrid or
                                                     # UnstructuredDisplacementField
rho0 = read_hdf5_background_density("field.h5")    # None, float, ndarray, or
                                                     # a DiscreteBackgroundDensity
```

`length_scale_m` rescales the stored coordinates (and the displacement
values, since they share the same length unit) to meters; use it when a file
was written in a different unit than meters. Both `displacement_m` and,
where a batch axis is present, individual snapshots may be `complex128` —
see [Complex outputs](../concepts/complex-outputs.md) for what a complex
displacement means physically.

## Background density

`read_hdf5_background_density` reads the companion `rho0_kg_m3` dataset (and,
if present, `grad_rho0_kg_m4`) separately from the field geometry. Depending
on what the file stores, it returns `None` (absent), a scalar `float`
(uniform background), a `DiscreteBackgroundDensity` (per-cell values on an
unstructured mesh), a `DifferentiableDiscreteBackgroundDensity` (per-cell
values plus a gradient), or an `np.ndarray` (per-cell values on a structured
grid). The `length_scale_m` you pass here must match the one used to import
the field, since density values and the coordinate scale are coupled.

## Batched snapshots

A file may carry a leading batch axis of size `P` on `displacement_m`
(`(P, ...)` — one snapshot per time, frequency, or realization index) rather
than a single field. This on-disk batch axis is a storage-layout convenience
for streaming I/O; it is not the same thing as the parametric axes in
gragra's in-memory API shapes (see [Arrays, dtypes and axes](../concepts/arrays.md)).

`iter_hdf5_field_snapshots` streams a batched file snapshot-by-snapshot
without materializing the full `(P, ...)` array, yielding a
`DisplacementGrid` or `UnstructuredDisplacementField` per requested index.
`from_hdf5_batched_weights` goes one step further and computes the batched
source weights directly, returning `(positions_m, weights)` with
`weights` shaped `(N, P)`.

## What the file does not carry

**gragra's importers do not read or interpret any coordinate-metadata
dataset the file may contain** — for example a `/time` dataset recorded
alongside a batched, time-sampled field. The batch axis is treated purely as
an opaque index; gragra has no time or frequency semantics anywhere in its
core (see [Chunking, softening and tolerances](../concepts/numerics.md) for
the related point about `gragra` never interpreting units other than SI
length, mass and time-independent kinematics). If you need to know which
wall-clock time or frequency a given snapshot index corresponds to, you must
track that yourself outside of gragra — the importer will not do it for you
and will not warn you that it isn't doing it.

## Bringing data from other solvers

**OpenFOAM has one bounded non-HDF5 reader.** With the `mesh` extra,
`load_openfoam_density` reads Foundation OpenFOAM 14 patch 20260724
reconstructed single-region legacy-ASCII `.vtk` density snapshots with native
`C`, `Vc`, and `cellID`, selecting `rho` or `delta_rho`. It is not a native
`polyMesh` reader, a general OpenFOAM reader, or a CFD solver. For other
OpenFOAM output, convert the data into one of the HDF5 layouts described on
this page and load it with `from_hdf5_field`; gragra makes no statement about
the correctness of that conversion.

The one solver gragra reads natively is SPECFEM3D Cartesian, and only in one
direction (solver output → gragra HDF5, not the reverse) — see
[SPECFEM3D Cartesian](specfem3d.md).

## Status and verification

### `gragra.adapters` HDF5 field I/O { #status-adapters-hdf5-field-io }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Experimental |
    | **Verified against** | invariance property (round-trip write/read on synthetic fixtures) |
    | **Agreement** | Exact round trip of synthetic array data (write with a schema-conformant writer, read back, compare arrays) |
    | **Test evidence** | `tests/test_hdf5_field_io.py::test_from_hdf5_unstructured_roundtrip`, `tests/test_hdf5_field_io.py::test_from_hdf5_displacement_grid_roundtrip`, `tests/test_hdf5_field_io.py::test_h1_batched_unstructured_roundtrip`, `tests/test_hdf5_field_io.py::test_h2_batched_structured_roundtrip`, `tests/test_hdf5_field_io.py::test_schema_violations` |
    | **Conditions** | Requires the `field-io` extra (`h5py`). Datasets must already conform to the schema described on this page — the reader validates shape and required keys but does not repair a malformed file |
    | **Not verified** | The correctness of any external tool's conversion into this schema — a file that validates against the schema is not thereby scientifically validated, and gragra's importer makes no statement about how faithfully an upstream solver's physics was captured |
    | **Known limits** | No `mesh_type="surface"` file schema exists yet. Coordinate metadata such as a `/time` dataset is never read or interpreted — see [What the file does not carry](#what-the-file-does-not-carry) |
    | **Scope** | The whole HDF5 field schema; it is not yet frozen |
