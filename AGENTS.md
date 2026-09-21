# Public repository boundary

This repository is the public distribution and verification lane for `gragra`.
It covers public candidate code and user-facing package quality, not private
research branches, private data, or arbitrary research environments.

Supported environments and environments actually tested by public CI are
distinct. CI success does not certify arbitrary scientific conditions. Run
research-required scientific verification locally with the exact code,
environment, inputs, and conditions identified for the work. GitHub Actions
success is not a universal adoption condition for research PRs. Automatic
Actions or environment matrices are not reintroduced without an explicit
policy change; moving execution location or frequency does not weaken
scientific acceptance conditions.

Public CI selects documentation-only checks for documentation-only changes.
Source, tests, build, dependency, workflow, script, and unknown changes select
the full public checks: Python 3.11/3.12 reference tests, Numba and C++ parity,
package wheel/sdist smoke tests in fresh environments, and a strict docs build.

Do not add private research material, persistent solver output, generated
artifacts, publication/deployment/upload steps, or release changes to this
lane. Keep optional-backend checks strict: an unavailable requested backend is
an error, not a NumPy fallback.
