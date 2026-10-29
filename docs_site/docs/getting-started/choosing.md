# Choosing extras and backends

## Which optional extras do I need?

The base installation includes NumPy, Astropy, pandas, and GWexpy.
The numerical functions still use NumPy arrays and internal types.
The extras below declare dependencies for specific capabilities or backends.
GWexpy may bring in some of those libraries transitively, but the extras
remain the explicit installation choices for those features.

| I want to... | Install |
|---|---|
| Load a surface or volume mesh from a file (STL, OBJ, ...) | `mesh` |
| Check an imported mesh for degenerate elements, non-manifold edges, etc. | `mesh-quality` |
| Run a large source count faster on CPU with JIT compilation | `numba` |
| Direction-average over a Lebedev quadrature grid | `lebedev` |
| Import an HDF5 displacement or density field (structured, unstructured, or batched) | `field-io` |

GWexpy field conversion is available with the base installation.
If you only need point sources, geometry primitives, and observables,
`pip install -e "."` is sufficient.

## Which compute backend should I use?

Start with the default, `"numpy"` — it is the reference implementation and
needs no extra install. Move to a different backend only once you have a
concrete reason:

- **Large source count, CPU only, no compiler available to you:** `"numba"`
  (needs the `numba` extra; pays a one-time JIT compilation cost on first
  call).
- **Large source count, CPU only, compiler available:** `"cpp"` (needs a
  source build with `GRAGRA_BUILD_CPP=ON`; OpenMP-threaded).
- **Very large source count, NVIDIA or AMD GPU available:** `"cupy"` (needs
  a matching CuPy wheel and a usable device; see the caveat in
  [Compute backends](../guide/backends.md) about this path never running in
  continuous integration).

None of these change what is computed, only how fast — see
[Compute backends](../guide/backends.md) for the full error contract and
the [Status and verification](../guide/backends.md#status-and-verification)
blocks for what each backend has actually been checked against.

## Decision table

| Situation | Extra / backend |
|---|---|
| Reading a `.stl`/`.obj` mesh | `mesh` |
| Auditing a mesh before using it as a source | `mesh-quality` |
| Fast CPU, no compiler | `numba` backend (needs `numba` extra) |
| Fast CPU, compiler available | `cpp` backend (source build, `GRAGRA_BUILD_CPP=ON`) |
| GPU available, large N | `cupy` backend (independent CuPy wheel install) |
| Exporting a SPECFEM3D Cartesian run | core install only (`gragra.adapters.specfem`, no extra) |
| Importing an HDF5 field written by that exporter, or any other HDF5 field | `field-io` |
| Interop with GWexpy field objects | base installation |
| Direction-averaged plane-wave statistics | `lebedev` |

## What none of the extras give you

No extra adds a general-purpose FEM/SEM solver, a general OpenFOAM reader, or
a way to run any backend without a real device or compiler present. The
`mesh` extra does add the bounded Foundation OpenFOAM 14 patch 20260724
legacy-ASCII `.vtk` density reader (`rho` or `delta_rho` with native
`C`/`Vc`/`cellID`), but not native `polyMesh`, other OpenFOAM formats, or a
CFD solver. The `field-io` extra provides `read_specfem_gll_hdf5` for its
documented narrow SPECFEM-derived HDF5 profile. The remaining absences are
documented — see
[Limitations and non-guarantees](../verification/limitations.md).
