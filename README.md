# gragra

`gragra` calculates Newtonian gravitational perturbations from discretised mass
distributions. This `v0.1.0rc2` documentation covers only a point-source smoke
path and a SPECFEM3D Cartesian GLL input path. Python 3.11 or later is required.

The reader accepts an existing canonical schema-v1 HDF5 file for a single-file,
time-sampled, ENU, elastic, single-material, non-PML input. It does not create
SPECFEM input or solver output.

## Public verification boundary

This repository verifies public candidate code and the quality of the published
package surface. It does not verify private research branches, private data, or
arbitrary research environments. Supported environments and environments
actually exercised by CI are distinct; CI success does not certify arbitrary
scientific conditions. Research-required scientific verification remains a
local responsibility and must be run with the code, environment, inputs, and
conditions identified for that work.

Public CI runs documentation-only checks for documentation changes. Changes to
source, tests, builds, dependencies, workflows, scripts, or unknown paths run
the full public checks: Python 3.11 and 3.12 reference tests, NumPy/Numba
parity, a real C++ build and parity check, package wheel and sdist smoke tests
in fresh environments, and a strict documentation build. These checks run on
public standard runners and do not imply a publication or deployment step.

## Install

`gragra` is not published on PyPI. Install a GitHub Release asset or the source
tag; see [Getting started](docs_site/docs/getting-started.md). Download
`SHA256SUMS` with the selected asset and verify that asset in their directory:

```console
grep '  gragra-0.1.0rc2-py3-none-any.whl$' SHA256SUMS | sha256sum --check -
```

Use the [GitHub Release](https://github.com/tatsuki-washimi/gragra/releases/tag/v0.1.0rc2)
or the [v0.1.0rc2 source tag](https://github.com/tatsuki-washimi/gragra/tree/v0.1.0rc2).

## Point-source smoke check

```python
import numpy as np
from gragra import G_SI, PointMass, point_acceleration

result = point_acceleration(PointMass(1.0, [-1.0, 0.0, 0.0]), [[0.0, 0.0, 0.0]])
np.testing.assert_allclose(result, [[-G_SI, 0.0, 0.0]])
assert result.shape == (1, 3)
assert not result.flags.writeable
```

## Reader example

The reader needs a canonical schema-v1 HDF5 file supplied by the caller.

```python
from gragra.adapters.specfem.reader import read_specfem_gll_hdf5
from gragra.observables import displacement_field_acceleration

dataset = read_specfem_gll_hdf5(
    "existing-canonical-gll.h5", profile_density_range=(1800.0, 3500.0)
)
acceleration = displacement_field_acceleration(dataset, [[100.0, 0.0, 0.0]])
```

Read the [numerical contract](docs_site/docs/contracts.md),
[SPECFEM3D profile](docs_site/docs/specfem3d-cartesian.md),
[backends](docs_site/docs/backends.md), [rc2 scope](docs_site/docs/rc2-scope.md),
and [verification limits](docs_site/docs/verification.md). See
[CONTRIBUTING.md](CONTRIBUTING.md) and [CITATION.cff](CITATION.cff).

## Reproduction material

This repository does not distribute persistent HDF5 fixtures, solver output, or
reproduction archives. CI creates small temporary HDF5 inputs for its checks.

## License

MIT. See [LICENSE](LICENSE).
