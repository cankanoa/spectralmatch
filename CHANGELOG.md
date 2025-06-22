# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## [v0.0.5](https://github.com/spectralmatch/spectralmatch/releases/tag/v0.0.5) - 2025-04-18

<small>[Compare with first commit](https://github.com/spectralmatch/spectralmatch/compare/e52b69e312b873db88e7267eeb89b80255fa30bf...v0.0.5)</small>

### Added

- Add GitHub Actions workflow to publish Python package ([35a6a40](https://github.com/spectralmatch/spectralmatch/commit/35a6a4096e17a885881a68659e9b5478fd4fdb71) by iosefa).
- Add Codecov badge to README.md ([db177f6](https://github.com/spectralmatch/spectralmatch/commit/db177f6719334f31c0620526dca45f4bb37e683b) by iosefa).
- Add Fiona to project dependencies ([1a72862](https://github.com/spectralmatch/spectralmatch/commit/1a72862c4087b36d94dbb4d18b8f8af31971ae9e) by iosefa).
- Add GitHub Actions workflow for build and test ([4260e43](https://github.com/spectralmatch/spectralmatch/commit/4260e4314bcd011f64afc4edaaf58442b4ab7974) by iosefa).
- Add initial test suite with pytest configuration ([66660c0](https://github.com/spectralmatch/spectralmatch/commit/66660c0d665fc442b137f0a62c7a2e967796bd1d) by iosefa).
- Add __all__ to spectralmatch module ([06da2a1](https://github.com/spectralmatch/spectralmatch/commit/06da2a1a27ac4cbba2cd81a70e1406e27eac60e3) by cankanoa).
- Add cloud mask generation and post-processing functions ([cfc4672](https://github.com/spectralmatch/spectralmatch/commit/cfc46727853137bcec45000b1deb9e98ee1ccf22) by cankanoa).
- Add vector mask support ([5f48027](https://github.com/spectralmatch/spectralmatch/commit/5f48027d9e7c3547edcfeeca639860845b59c1aa) by cankanoa).
- Add example starting image, update script comments , update .gitignore ([b3f1e08](https://github.com/spectralmatch/spectralmatch/commit/b3f1e0831cb654e7d74794134236def4fc0a8862) by cankanoa).
- Add test images for WorldView-3 dataset ([83563b3](https://github.com/spectralmatch/spectralmatch/commit/83563b33b0300c421bbb242e95e05721c44544aa) by cankanoa).
- Add utility to align, mask, and merge images. ([b4c1259](https://github.com/spectralmatch/spectralmatch/commit/b4c125921b790e4f46ebc1786b50d4eb269228fe) by cankanoa).
- Add debug mode and custom projection support ([0ffaa72](https://github.com/spectralmatch/spectralmatch/commit/0ffaa72bf744221ff9a3c8bae0a96abf043c3237) by cankanoa).
- Add main script for automated image mosaicing ([1134223](https://github.com/spectralmatch/spectralmatch/commit/1134223c125efeb98a98046218b54cc3e2055572) by cankanoa).
- Add initial implementations for global and local histogram matching ([3c4209e](https://github.com/spectralmatch/spectralmatch/commit/3c4209e55e4469ff6c50dde0147d8dda3ea611d2) by cankanoa).

### Fixed

- Fix typo in Conda command for environment creation. ([b9e33e7](https://github.com/spectralmatch/spectralmatch/commit/b9e33e7c4690ce10784b3a75aff44d8a340c03e2) by iosefa).
- Fix typos in README.md for clarity and accuracy ([7bf76ec](https://github.com/spectralmatch/spectralmatch/commit/7bf76ec55eebfa7ebbcde352632f681b299ff66f) by cankanoa).

### Removed

- Remove not yet supported imports from `__init__.py` ([41099b9](https://github.com/spectralmatch/spectralmatch/commit/41099b99b9d37b933319c3bf11634bba5705ac3a) by cankanoa).

