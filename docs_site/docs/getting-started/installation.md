# Installation

Install gragra from a source checkout for local research and development.

## Requirements

Python ≥3.11. The base installation requires `numpy>=1.24.0`,
`astropy>=7.2,<9`, `pandas>=2.2.3,<4`, and `gwexpy>=0.2.3,<0.3`.
The numerical layer still uses NumPy arrays and internal types and does not
import Astropy, pandas, or GWexpy.
GWexpy may install SciPy, h5py, Matplotlib, and other libraries transitively;
these are not direct gragra requirements unless an optional feature declares them.

## Installing from source

```bash
git clone https://github.com/tatsuki-washimi/gragra.git
cd gragra
pip install -e "."
```

For development (running the test suite, building this documentation site,
contributing a change), install the `dev` extra instead:

```bash
pip install -e ".[dev]"
```

`pip install -e "."` gives you point sources, geometry primitives,
observables, and the GWexpy bridge. Mesh import, HDF5 field I/O, and the
directional quadrature grid have separate optional extras, described next and in
[Choosing extras and backends](choosing.md).

## Optional extras

| Extra | Adds |
|---|---|
| `mesh` | `meshio`-based surface/volume mesh loading |
| `mesh-quality` | mesh quality diagnostics for imported meshes |
| `numba` | the JIT-compiled `"numba"` compute backend |
| `lebedev` | the Lebedev directional quadrature grid |
| `field-io` | `h5py`-based HDF5 field import (structured, unstructured, batched) |

Install any combination with `pip install -e ".[mesh,field-io]"` and so on.
These extras declare the dependencies needed for each feature even when a
base dependency happens to bring one in transitively.

## Optional compiled and GPU backends

Two further backends are not extras in the `pip install -e ".[...]"` sense,
because they involve a compiled artifact rather than a pure-Python
dependency:

- **The C++/OpenMP backend** (`backend="cpp"`) requires a source build with
  the compiled extension turned on:
  `pip install -e "." --config-settings=cmake.define.GRAGRA_BUILD_CPP=ON`,
  and a C++17 compiler. A plain `pip install -e "."` does not build it, and
  `backend="cpp"` then raises `ImportError` at call time rather than
  silently falling back — see [Compute backends](../guide/backends.md).
- **The CuPy backend** (`backend="cupy"`) requires a CuPy wheel matching
  your local CUDA or ROCm runtime, installed independently of gragra (CuPy's
  own wheel selection depends on your driver version, which gragra cannot
  determine for you), and a usable GPU device at call time.

Neither backend is required for correctness — the default `"numpy"` backend
is the reference implementation every other backend is checked against; see
[Compute backends](../guide/backends.md).

## Verifying the installation

```python
--8<-- "snippets/quickstart_verify_install.py"
```

**Common installation failures:**

- An optional extra's import fails at runtime (`ImportError: h5py is
  required...`) — you called a function that needs an extra you did not
  install; see [Choosing extras and backends](choosing.md) for which
  function needs which extra.
- The C++ backend does not build — building it is opt-in and needs a C++17
  compiler; see [Troubleshooting](../help/troubleshooting.md).
