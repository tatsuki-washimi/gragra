# gragra

`gragra` calculates Newtonian gravitational perturbations from discretised mass distributions.

## SPECFEM3D Cartesian GLL input (rc1)

The release candidate accepts a single-file, `time_sampled` HDF5 snapshot written by `gragra.adapters.specfem.writer`. It supports elastic, single-material, non-PML `NGNOD=8` data. The reader validates the provenance schema, on-disk dtypes, array shapes, cadence and density range before returning an immutable in-memory snapshot.

`displacement_field_acceleration` evaluates one time sample at a time after loading the snapshot. It is not an HDF5 streaming reader. NumPy is the reference backend; Numba and the optional C++ backend are checked for parity in CI.

```python
from gragra.adapters.specfem.reader import read_specfem_gll_hdf5
from gragra.observables import displacement_field_acceleration

dataset = read_specfem_gll_hdf5("gll.h5", profile_density_range=(1800.0, 3500.0))
acceleration = displacement_field_acceleration(dataset, [[100.0, 0.0, 0.0]])
```

## Reproduction

Install `.[dev,field-io]` and run `pytest -q tests/test_specfem_reader.py`. The test generates a complete synthetic HDF5 fixture with the public writer and verifies the reader, mutation rejection, independent gravity oracle and backend contract.

## License

MIT. See [LICENSE](LICENSE).
