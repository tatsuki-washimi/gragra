# Backends

NumPy is the canonical rc2 backend. Numba and C++ are checked in CI against
NumPy on a small HDF5 input created temporarily by the test job, with
`rtol=1e-13`. That is the extent of the public backend-parity claim.

The published wheel is pure Python and has the C++ backend disabled. CuPy,
performance claims, and parity on real solver output are outside the rc2 public
verification scope.

The documented rc2 surface is:

- `read_specfem_gll_hdf5(path, *, profile_density_range)`
- `SpecfemGLLDataset` and `iter_blocks()`
- `displacement_field_acceleration(dataset, targets, *, chunk_size=None, backend="numpy")`
- `PointMass`, `G_SI`, and `point_acceleration()` for the smoke check

Other APIs may be bundled in the distribution, but they are not covered by the
rc2 public verification promise.
