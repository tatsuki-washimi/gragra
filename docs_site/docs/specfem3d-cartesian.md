# SPECFEM3D Cartesian GLL input

`read_specfem_gll_hdf5(path, *, profile_density_range)` reads canonical
schema-v1 HDF5 data into an immutable `SpecfemGLLDataset`.

The intended solver profile is external-mesh, `xdecompose_mesh`, `NGNOD=8`, and
a standard NGLL configuration. `NGNOD=8` and standard NGLL are profile
assumptions; the reader does not independently recover or verify `NGNOD=8` from
HDF5.

The reader checks schema v1; `single_file`; `time_sampled`; ENU coordinates; a
single elastic domain; a single material; no PML elements; finite positive
density and quadrature volume; `field_units.displacement == "m"`; required
dataset dtypes, shapes, and time relations; internally consistent provenance
and cadence metadata; and element ID values. It checks that element IDs are
ascending `0..N-1` and interprets the GLL axis in its stated contract order; it
does not independently establish a physical GLL flattening order from HDF5. It
rejects any PML element; it does not remove PML data.

A finite `profile_density_range=(low, high)` checks density inclusively at both
endpoints. `profile_density_range=None` intentionally omits only that external
profile-range check. It still requires density to be finite and positive.

Cadence and provenance checks establish internal metadata consistency. They do
not independently certify a solver run or physical sampling choice.

`SpecfemGLLDataset.iter_blocks()` yields contiguous element-ID blocks without
combining duplicate coordinates. Its point mass is `density * quadrature_volume`.
