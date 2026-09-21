# Contributing

Please open an issue before proposing a substantial change. Contributions must
include focused tests and must not add private research material, solver output
or third-party source text without a documented redistribution permission.

## Verification responsibility

The public repository covers public candidate code and user-facing package
quality. It does not cover private research branches or private data. The CI
workflow selects documentation-only checks for documentation changes and full
public-scope checks for source, tests, builds, dependencies, workflows,
scripts, and unknown paths. Full checks include the declared Python 3.11 and
3.12 environments, NumPy/Numba and C++ parity, package wheel/sdist smoke
tests, and a strict docs build.

Supported environments are not the same as environments actually tested by CI.
CI success does not certify arbitrary scientific conditions. Run research-
required scientific verification locally with the exact code, environment,
inputs, and conditions relevant to the claim, and record those conditions in
the change description or accompanying research log. GitHub Actions success is
not a universal adoption condition for research PRs. Automatic Actions or
environment matrices are not reintroduced without an explicit policy change;
moving execution location or frequency does not weaken scientific acceptance
conditions.

The workflow performs no publication, deployment, or upload operation.
