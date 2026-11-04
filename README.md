<p align="center">
  <a href="https://tatsuki-washimi.github.io/gragra/">
    <img src="docs_site/docs/assets/brand/gragra-apple-red.svg" alt="gragra logo" width="88">
  </a>
</p>

# gragra

**Newtonian gravitational perturbation calculations in Python.**

`gragra` computes gravitational potential, acceleration, and gravity-gradient
responses from prescribed mass distributions, density perturbations, and moving
or displaced sources. Applications include Newtonian-noise modelling for
gravitational-wave detectors. It provides analytic models as well as numerical
calculation paths; it does not solve seismic or fluid-dynamics equations.

**[Documentation](https://tatsuki-washimi.github.io/gragra/)** ·
[Model & method catalogue](https://tatsuki-washimi.github.io/gragra/guide/model-and-method-catalogue/) ·
[API reference](https://tatsuki-washimi.github.io/gragra/api/) ·
[Verification & limitations](https://tatsuki-washimi.github.io/gragra/verification/)

## Installation

Python **3.11 or later** is required. `gragra` is not published on PyPI.

To install the current public source tree:

```bash
git clone https://github.com/tatsuki-washimi/gragra.git
cd gragra
python -m pip install -e .
```

For the published pre-release and its downloadable packages, see
[GitHub Releases](https://github.com/tatsuki-washimi/gragra/releases).

See the [installation guide](https://tatsuki-washimi.github.io/gragra/getting-started/installation/)
for optional extras, compute backends, and verification of the installation.
The [minimal example](https://tatsuki-washimi.github.io/gragra/getting-started/minimal-example/)
shows a first calculation.

## Updates

- **Documentation (November 2026):** 24 physical-model walkthroughs with
  runnable STEP 1–4 examples and 31 calculation figures are available on the
  [documentation site](https://tatsuki-washimi.github.io/gragra/).
- **Current public `main`:** package version `0.2.0rc1`; its GitHub Release
  is currently a draft.
- **Latest published pre-release:** [`v0.1.0rc2`](https://github.com/tatsuki-washimi/gragra/releases/tag/v0.1.0rc2)
  (6 August 2026).

For detailed changes and future releases, consult the
[Changelog](CHANGELOG.md) and [GitHub Releases](https://github.com/tatsuki-washimi/gragra/releases).

## Further information

The documentation covers
[physical models](https://tatsuki-washimi.github.io/gragra/guide/physics-reference/),
[numerical methods](https://tatsuki-washimi.github.io/gragra/guide/numerical-methods/),
[API usage](https://tatsuki-washimi.github.io/gragra/api/), and
[verification scope and limitations](https://tatsuki-washimi.github.io/gragra/verification/limitations/).
Public CI checks are scoped to the public package; research-required scientific
verification depends on the exact code, environment, inputs, and conditions.

[Contributing](CONTRIBUTING.md) · [Citation](CITATION.cff) · [MIT License](LICENSE)
