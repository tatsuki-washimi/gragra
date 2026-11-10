<p align="center">
  <a href="https://tatsuki-washimi.github.io/gragra/">
    <img
      src="docs_site/docs/assets/branding/gragra-logo-horizontal.png"
      alt="GraGra logo"
      width="640"
    >
  </a>
</p>

# gragra

**Newtonian gravitational perturbation calculations in Python.**

`gragra` computes gravitational potential, acceleration, and gravity-gradient
responses from prescribed mass distributions, density perturbations, moving
sources, and displacement fields.

Its main application is **Newtonian-noise modelling for gravitational-wave
detectors**, but it also provides general tools for forward calculations of
gravity perturbations from analytic models, discretised geometries, and external
field data.

For the full documentation, physical models, API details, and verification
scope, see the public documentation site:

- **Documentation:** https://tatsuki-washimi.github.io/gragra/
- **Physics reference:** https://tatsuki-washimi.github.io/gragra/guide/physics-reference/
- **Model & method catalogue:** https://tatsuki-washimi.github.io/gragra/guide/model-and-method-catalogue/
- **API reference:** https://tatsuki-washimi.github.io/gragra/api/
- **Verification & limitations:** https://tatsuki-washimi.github.io/gragra/verification/

## Installation

Python **3.11 or later** is required.

For PyPI distribution availability and installation, see
https://pypi.org/project/gragra/. Before a version is published there, install
from the public source repository as shown below.

To install the 0.2.0rc1 pre-release **after it appears on PyPI**:

```bash
python -m pip install gragra==0.2.0rc1
```

### Install from source

```bash
git clone https://github.com/tatsuki-washimi/gragra.git
cd gragra
python -m pip install -e .
```

### Install with development tools

If you want to run tests, build documentation, or contribute changes:

```bash
python -m pip install -e ".[dev]"
```

### Optional features

Optional extras are available for specific workflows:

- `mesh` — mesh import support
- `mesh-quality` — mesh quality diagnostics
- `numba` — JIT backend
- `lebedev` — Lebedev directional quadrature
- `field-io` — HDF5 field import

Example:

```bash
python -m pip install -e ".[mesh,field-io]"
```

For compiled and GPU backends, installation verification, and troubleshooting,
see the installation guide:

- **Installation guide:** https://tatsuki-washimi.github.io/gragra/getting-started/installation/
- **Minimal example:** https://tatsuki-washimi.github.io/gragra/getting-started/minimal-example/

## Updates

### Documentation update (November 2026)

The public documentation site now includes:

- **24 physical-model detail pages**
- **STEP 1–4 walkthroughs** for all published models
- **31 deterministic calculation figures**
- expanded **physics reference**, **numerical methods**, and **API reference**

See the documentation site for the current published content:

https://tatsuki-washimi.github.io/gragra/

### Release status

- The current public `main` branch carries package version **`0.2.0rc1`**
- Consult the [PyPI project page](https://pypi.org/project/gragra/) for the current PyPI release status.
- Consult [GitHub Releases](https://github.com/tatsuki-washimi/gragra/releases) for available GitHub release assets and draft/publication status.

For detailed change history, see:

- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **Releases:** https://github.com/tatsuki-washimi/gragra/releases

## What gragra provides

`gragra` provides:

- forward calculations of **gravitational potential**
- **acceleration** and **gravity-gradient tensor** observables
- analytic and discretised source models
- support for displacement-field and density-perturbation inputs
- documented public APIs and reproducible examples

It does **not** act as a general PDE solver for elastodynamics or fluid
dynamics. Instead, it evaluates gravity perturbations from sources or fields
supplied analytically or by the caller.

## Further information

- [Documentation site](https://tatsuki-washimi.github.io/gragra/)
- [Contributing](https://tatsuki-washimi.github.io/gragra/help/contributing/)
- [Citation](CITATION.cff)
- [License (MIT)](LICENSE)

## Snippet-backed examples

```python
--8<-- "snippets/readme_point_mass_acceleration.py"
```

```python
--8<-- "snippets/readme_coherent_response.py"
```

```python
--8<-- "snippets/readme_continuous_sources.py"
```
