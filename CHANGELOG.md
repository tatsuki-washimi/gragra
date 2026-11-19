# Changelog

All notable changes to this project are documented in this file.

## [0.2.0] - Unreleased

The release date is set when 0.2.0 is published.

First stable release of the 0.2 series. The library code is the same as in
0.2.0rc1; this release makes no API or numerical-behaviour change. The
differences from 0.2.0rc1 are the package version and metadata, release
documentation, wording in the documentation and in the docstring of
`displacement_field_acceleration` that referred to the release candidate, and
the PyPI publication workflow.

For the contents of the 0.2 series and the migration from 0.1.0rc2, see the
0.2.0rc1 entry below.

## [0.2.0rc1] - 2026-11-11

### Added

- R2 analytic formulas, R6 trajectory and rigid motion, R3 plane-wave and field
  evaluation paths, and R4 material-region / hex8 geometry support from the
  development line (source revision `b19ccbb`).
- Expanded documentation site, examples, and public regression tests aligned
  with the above API surface.

### Changed

- Package version and dependencies aligned with the development line (Astropy,
  pandas, GWexpy, SciPy, Matplotlib, optional backends unchanged in role).
- Root public API follows `gragra.gravity_sources` and consolidated observables;
  rc2 `gragra.sources` and root-level projection helpers are removed (breaking).

### Removed

- `water_channel_uncorrelated_chen_taup2019_displacement_asd` from the public
  API (independent uncorrelated channel models remain).

### Migration from 0.1.0rc2

- Replace `gragra.sources` imports with `gragra.gravity_sources`.
- Replace removed root projection helpers with `gragra.observables` entry points
  documented in the API reference.
