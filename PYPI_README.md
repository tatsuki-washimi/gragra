# gragra

**Newtonian gravitational perturbation calculations in Python.**

gragra computes Newtonian gravitational potential, acceleration, and gravity-gradient tensor observables from prescribed mass distributions, density perturbations, moving sources, and displacement fields. Its principal application is Newtonian-noise modelling for gravitational-wave detectors.

## Install

Requires **Python 3.11 or newer**.

```bash
python -m pip install gragra==0.2.0rc1
```

Version 0.2.0rc1 is a **pre-release**. See the [model and method catalogue](https://tatsuki-washimi.github.io/gragra/guide/model-and-method-catalogue/) and [verification and limitations](https://tatsuki-washimi.github.io/gragra/verification/) for documented applicability conditions.

## Minimal example

```python
import numpy as np
from gragra import PointMass, point_acceleration

point = PointMass(mass_kg=1.0, position_m=[0.0, 0.0, 0.0])
acc = point_acceleration(point, [[1.0, 0.0, 0.0]])
print(acc)  # acceleration vector, m/s^2
```

The default NumPy backend is available without compiled extensions. Optional features include mesh I/O, HDF5 field input, Numba, and C++/GPU backends subject to their documented dependencies and environment requirements.

## Documentation and development

- [Documentation and tutorials](https://tatsuki-washimi.github.io/gragra/)
- [Physics reference](https://tatsuki-washimi.github.io/gragra/guide/physics-reference/)
- [24-model catalogue and verification scope](https://tatsuki-washimi.github.io/gragra/guide/model-and-method-catalogue/)
- [API reference](https://tatsuki-washimi.github.io/gragra/api/)
- [Public source code and issues](https://github.com/tatsuki-washimi/gragra)
- [Changelog](https://github.com/tatsuki-washimi/gragra/blob/main/CHANGELOG.md)

gragra evaluates gravity perturbations from prescribed sources and fields. It is not a general elastodynamic or fluid-dynamic PDE solver.

MIT License. Maintainer: Tatsuki Washimi.
