# Getting started

## Install a published release asset

`gragra` is not on PyPI. Download the pure-Python wheel and `SHA256SUMS` from
the [v0.1.0rc2 GitHub Release](https://github.com/tatsuki-washimi/gragra/releases/tag/v0.1.0rc2),
then verify and install them in a clean virtual environment.

```console
python -m venv .venv
. .venv/bin/activate
grep '  gragra-0.1.0rc2-py3-none-any.whl$' SHA256SUMS | sha256sum --check -
python -m pip install "h5py>=3.11,<3.12" gragra-0.1.0rc2-py3-none-any.whl
```

The wheel is pure Python and has the C++ backend disabled.

## Install from the source tag

```console
git clone --branch v0.1.0rc2 --depth 1 https://github.com/tatsuki-washimi/gragra.git
cd gragra
python -m venv .venv
. .venv/bin/activate
python -m pip install ".[field-io]"
```

## Check a point mass

```python
import numpy as np
from gragra import G_SI, PointMass, point_acceleration

result = point_acceleration(PointMass(1.0, [-1.0, 0.0, 0.0]), [[0.0, 0.0, 0.0]])
np.testing.assert_allclose(result, [[-G_SI, 0.0, 0.0]])
assert result.shape == (1, 3)
assert not result.flags.writeable
```

## Read an existing GLL file

This syntactically checked example requires a canonical schema-v1 HDF5 file
supplied by the caller at runtime. The package does not provide a solver run or
an example file.

```python
from gragra.adapters.specfem.reader import read_specfem_gll_hdf5

dataset = read_specfem_gll_hdf5(
    "existing-canonical-gll.h5", profile_density_range=(1800.0, 3500.0)
)
```

See [SPECFEM3D Cartesian](specfem3d-cartesian.md) for the accepted profile.
