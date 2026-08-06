# SPECFEM3D Cartesian GLL

`read_specfem_gll_hdf5` accepts the schema-backed, single-file `time_sampled` format produced by gragra's writer. The candidate supports elastic, single-material, non-PML `NGNOD=8` data only.

The reader validates all required provenance, exact HDF5 dtypes and array shapes before closing the file and returning immutable arrays. The acceleration observable processes the retained snapshots one time index at a time.

Density point masses are `density * quadrature_volume`. Coincident GLL coordinates from different elements are deliberately not merged.
