# Change Log

## [2.2.1] - 2026-09-26

### Fixed

- Updated dependencies to address outstanding security advisories. Notable runtime updates are Pillow 12.3.0, urllib3 2.8.0, Tornado 6.5.10, asteval 1.0.10, and Requests 2.34.2
- Raised the minimum supported versions of `bokeh` to 3.8.2, `black` to 26.3.1, `mkdocs-material` to 9.7.7, and `pytest` to 9.0.3, so that a new install cannot resolve to a version with a known advisory

### Changed

- Continuous integration now installs dependencies from `uv.lock` with `uv sync --locked`, so tests run against the versions that are committed to the repository rather than whatever is newest on PyPI at the time
- Added a Dependabot configuration for the `uv` ecosystem, so future updates arrive as reviewable pull requests

## [2.2.0] - 2026-09-26

### Added

- Support for CCPNMRv3 (a3) peak lists that omit the unit suffix in the linewidth columns, i.e. `LW F1`/`LW F2` in addition to `LW F1 (Hz)`/`LW F2 (Hz)`. Current versions of CCPNMRv3 export the former

## [2.1.2] - 2025-09-16

### Fixed

- Bug in `peakipy edit` where peak markers were drawn as visible circles in the peak picking figure. They are now drawn with a near zero radius so that they are invisible but can still be selected

## [2.1.1] - 2025-09-15

### Changed

- Migrated from Poetry to uv for dependency management and updated all dependencies to their latest compatible versions. No functional changes

## [2.1.0] - 2025-02-23

### Added

- Support for `.csv` files that contain minimum of `ASS` (peak assignment),`X_PPM` (position of peak on x axis in parts per million) and `Y_PPM` (position of peak on y axis in parts per million) columns 
- Check for validity of radii for fitting masks (`--x-radius-ppm` and `--y-radius-ppm` must correspond to at least 2 points each)
- Scientific notation for Amplitudes and Heights
- Improved docs