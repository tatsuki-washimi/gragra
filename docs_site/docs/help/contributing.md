# Contributing and security

## Reporting a problem

Use the
[issue templates](https://github.com/tatsuki-washimi/gragra/tree/main/.github/ISSUE_TEMPLATE/).
A bug report needs enough to reproduce it: OS, Python version, gragra
version (or commit SHA — see
[Reproducibility and citation](citation.md)), backend, install method, a
minimal repro, input shapes, and the expected value with a tolerance. A
report about scientific correctness (a suspected wrong sign, a suspected
wrong closed-form comparison) should say which
[Verification](../verification/index.md) claim it contradicts, not just
that a number "looks wrong".

## Reporting a security issue

Do not open an issue containing sensitive security details. See
[SECURITY.md](https://github.com/tatsuki-washimi/gragra/blob/main/SECURITY.md)
for the reporting channel.
SECURITY.md's own text is authoritative if this page and it ever disagree.

## Development setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

```bash
pytest                                            # default (fast) suite
pytest -m slow                                    # convergence/benchmark tests
pytest --cov=gragra --cov-report=term-missing     # with coverage
```

Run `ruff check .` and `ruff format --check .` locally after source edits.

## What continuous integration actually checks

The CI workflow runs the NumPy core, optional C++ backend, optional extras,
and the local documentation build. Extra-specific jobs use
`GRAGRA_REQUIRE_EXTRAS=1` so a missing dependency cannot silently skip the
tests for that extra.
There is no GPU runner: nothing in CI ever exercises the `cupy` backend,
which is why [Compute backends](../guide/backends.md) states CuPy
equivalence as verified on the maintainer's local GPU only.

## Documentation conventions

Analytic cross-validation against a closed-form solution is this project's
primary test strategy; finite-difference checks are auxiliary (see
[Analytic benchmarks](../guide/analytic-benchmarks.md)). New public APIs
should follow that pattern where a closed form exists. Executable examples
live under `docs_site/snippets/` where practical; optional-dependency examples
may remain illustrative and are marked accordingly.

## Licensing and third-party material

gragra is MIT-licensed
([LICENSE](https://github.com/tatsuki-washimi/gragra/blob/main/LICENSE)).
Third-party material used by this documentation site, including MathJax, is
attributed in
[`docs/legal/third_party_notices.md`](https://github.com/tatsuki-washimi/gragra/blob/main/docs/legal/third_party_notices.md).
