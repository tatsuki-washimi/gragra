# Changelog

All notable changes to this project are documented in this file.

## [0.2.0rc1] - 2026-11-08

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
