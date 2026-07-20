# Changelog

All notable user-facing changes to `py_did_multiplegt_dyn` are documented here.

## 0.1.8 - 2026-07-18

### Added

- `reset`: partitions original groups into treatment-spell subgroups after a
  configurable number of unchanged periods, while retaining original-group
  clustering by default.
- `avg_time_periods`: reports the Stata-compatible average number of periods
  over which incremental treatment doses accumulate and the number of
  switchers contributing to each effect.
- Public result keys `avg_cumul` and `N_switch_avg_1`, `N_switch_avg_2`, etc.
- Unit and end-to-end tests for both options, including `by_path`
  compatibility.

### Changed

- The older inactive Python `avg_periods` calculation is retained only as a
  commented legacy reference; the supported calculation now follows the Stata
  implementation's switcher-specific valid-control horizons.
- The README now uses the 2026 estimator description as its authoritative
  reference, documents the Python-supported option set, and omits options that
  are not available through the Python estimator.
- Release automation now tests Python 3.8 and 3.12, verifies that the GitHub
  release tag matches package metadata, validates distributions with Twine,
  and publishes only after all checks pass.

### Packaging

- Package version advanced to `0.1.8`.
- Declared `pyarrow`, which is required by the package's Polars-to-pandas
  conversions.
- Added standard package and license metadata, optional test and release
  dependencies, and repository hygiene rules.
