# Change Log

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