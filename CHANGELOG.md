# Change Log

## [2.1.0] - 2025-02-23

### Added

- Support for `.csv` files that contain minimum of `ASS` (peak assignment),`X_PPM` (position of peak on x axis in parts per million) and `Y_PPM` (position of peak on y axis in parts per million) columns 
- Check for validity of radii for fitting masks (`--x-radius-ppm` and `--y-radius-ppm` must correspond to at least 2 points each)
- Scientific notation for Amplitudes and Heights
- Improved docs