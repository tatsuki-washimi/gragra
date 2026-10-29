# Third-party notices

gragra is distributed under the MIT License; see [LICENSE](LICENSE).

## Runtime dependencies

Third-party Python packages (NumPy, SciPy, Astropy, pandas, GWexpy, Matplotlib,
and their transitive dependencies such as gwpy) are subject to their respective
upstream licenses. GWexpy and gwpy are commonly used under MIT-style terms;
verify the license text shipped with your installed environment.

Matplotlib is a declared runtime dependency for compatibility with modules that
may import it; `gragra.plotting` is not part of the documented public API for
release 0.2.0rc1.

## Build and optional components

- **pybind11** (BSD 3-Clause) — optional C++ extension build.
- **h5py** (BSD 3-Clause) — optional field I/O extra.
- **Numba** (BSD 2-Clause) — optional acceleration extra.

## Documentation site (optional `docs` extra)

Local documentation builds may vendor **MathJax 3.2.2** (Apache License 2.0).
See `docs_site/docs/javascripts/mathjax/LICENSE.txt` when present.

Material for MkDocs, mkdocstrings, PyMdown Extensions, and related build tools
retain their upstream licenses in the documentation build environment.

## Scientific attribution

Some docstrings and documentation cite published papers and standard references.
Those citations describe provenance of formulas; they do not re-distribute the
underlying publications.
