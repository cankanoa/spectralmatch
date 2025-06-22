# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## [v1.0.0](https://github.com/spectralmatch/spectralmatch/releases/tag/v1.0.0) - 2025-06-22

<small>[Compare with v0.0.5](https://github.com/spectralmatch/spectralmatch/compare/v0.0.5...v1.0.0)</small>

### Added

- Add QGIS plugin improvements and dependency adjustments ([cbef567](https://github.com/spectralmatch/spectralmatch/commit/cbef567b85b25aafdd31c3afae557b2d91e56e15) by cankanoa).
- Add CLI tests and improve error handling and documentation ([d66d4f2](https://github.com/spectralmatch/spectralmatch/commit/d66d4f295500ca8566b19af2566ed26e3a3e212e) by cankanoa).
- Add 'create' target for plugin packaging with temp commit ([3b8fa4f](https://github.com/spectralmatch/spectralmatch/commit/3b8fa4f9aede25fc8dcccdc0e71e9c79f60ee496) by cankanoa).
- Add CLI support and documentation (#44) ([263ee51](https://github.com/spectralmatch/spectralmatch/commit/263ee5127b4d6a227471904312b418fee37bcf0c) by Kanoa Lindiwe).
- Add initial QGIS plugin ([4f4cd23](https://github.com/spectralmatch/spectralmatch/commit/4f4cd23e4a45f566f1c377afa9177ac9048a1f63) by cankanoa).
- Add extensive test coverage for spectralmatch user facing functions (#43) ([31f666a](https://github.com/spectralmatch/spectralmatch/commit/31f666a420bd2a8e28a8f559b9531249bcdd1c09) by Kanoa Lindiwe).
- Add details for output dtype and nodata value resolution ([17b7929](https://github.com/spectralmatch/spectralmatch/commit/17b79298b28cd0dc9d3e1bb0598bef516b073876) by cankanoa).
- Add support for saving outputs as Cloud Optimized GeoTIFF (COG) (#41) ([0ef6c06](https://github.com/spectralmatch/spectralmatch/commit/0ef6c0645984ce37fe92ee941de3831956d5ee1b) by Kanoa Lindiwe).
- Add automated LLM prompt generation and integration (#40) ([c4e43aa](https://github.com/spectralmatch/spectralmatch/commit/c4e43aa71370ec55180592ba203789dc3045d960) by Kanoa Lindiwe).
- Add custom band math function and minor documentation fixes ([bf6453b](https://github.com/spectralmatch/spectralmatch/commit/bf6453b54d61e59f218ae15b68fc5db40c8236a2) by cankanoa).
- Add initialization and logging improvements for raster merging ([b0e4154](https://github.com/spectralmatch/spectralmatch/commit/b0e4154af86361b0bf204fd19d2dafb394704143) by cankanoa).
- Add new features and improve documentation structure ([504e891](https://github.com/spectralmatch/spectralmatch/commit/504e89164e42ea668134bbf70d37d62fc17ad0f4) by cankanoa).
- Add validation utilities and universal/match type definitions ([afffaf7](https://github.com/spectralmatch/spectralmatch/commit/afffaf7ebd1f452ee9e2f0ace1b468b6f4e8ad4e) by cankanoa).
- Add save_as_cog option and enhance parallel processing ([512fae1](https://github.com/spectralmatch/spectralmatch/commit/512fae19afa2843c4beaaf862ce66165ba39d19e) by cankanoa).
- Add Markov Random Field-based seamline generation module ([53c27c1](https://github.com/spectralmatch/spectralmatch/commit/53c27c150fa80afefc14e0a1fc7980c9d62da459) by cankanoa).
- Add cutline output and improve EMP cutting logic ([de3736e](https://github.com/spectralmatch/spectralmatch/commit/de3736edebec47ff4150c691b41e4ad5655af9e9) by cankanoa).
- Add new methods and update details in RRN documentation ([6c3180e](https://github.com/spectralmatch/spectralmatch/commit/6c3180ec7ec5d1e7c7b9020af7c5361222b3a6ef) by cankanoa).
- Add documentation on RRN methods and update navigation ([fa8edd3](https://github.com/spectralmatch/spectralmatch/commit/fa8edd3885df6ac060ebc33ba3a8e62c53bdea9a) by cankanoa).
- Add full support for complex vector masks to withhold pixels from global and local matching. ([a9b14c0](https://github.com/spectralmatch/spectralmatch/commit/a9b14c091991eda5ed28c46648fa321cc595583f) by cankanoa).
- Add documentation for file formats and input requirements ([1533e60](https://github.com/spectralmatch/spectralmatch/commit/1533e6045b86618121a486773e9538ba0e9a1345) by cankanoa).
- Added feature to exclude images from model build process. Added feature to save and load image stats for reprocessing. Improved input and output params to accept folder or list of paths. ([821d247](https://github.com/spectralmatch/spectralmatch/commit/821d24761a02603d244d107264379abb48ebc2e7) by cankanoa).
- Add debug logging for raster masking function ([50ba5c0](https://github.com/spectralmatch/spectralmatch/commit/50ba5c052efeee3606a8ce9f7d7e9794aa390b28) by cankanoa).
- Add save support for local match, convert examples to Jupyter notebooks, and refactor debug mode. ([a233cbf](https://github.com/spectralmatch/spectralmatch/commit/a233cbff89ad2ac9da3c1f1d3e228b34b5c83a31) by cankanoa).
- Add error handling with traceback in _compute_tile_local ([79d1fa2](https://github.com/spectralmatch/spectralmatch/commit/79d1fa2c3cb094dbfc84ce9bbb3e190d61df57ca) by cankanoa).
- Add Voronoi-based seamline generation and implementation example ([224b22d](https://github.com/spectralmatch/spectralmatch/commit/224b22d5ecdff9db1436cdb0f304114aa0cbfe36) by cankanoa).
- Add return type annotations to several functions ([13fcc77](https://github.com/spectralmatch/spectralmatch/commit/13fcc774861fc0f20dfdede193286ff4423c8652) by cankanoa).
- Add automated setup and update documentation ([b11cae2](https://github.com/spectralmatch/spectralmatch/commit/b11cae21d7569031bfb03b63aff78a9d77924e49) by cankanoa).
- Add development and installation guides to documentation ([5d29782](https://github.com/spectralmatch/spectralmatch/commit/5d297823fdcc443c29be5b77311a04690b3bffda) by cankanoa).
- Add and update docstrings and param types for all functions. ([630b0be](https://github.com/spectralmatch/spectralmatch/commit/630b0bea87109d2e59e22af98bb804a3a5667a96) by cankanoa).
- Add Makefile, enhance docs, refactor code, and update examples ([91f87a6](https://github.com/spectralmatch/spectralmatch/commit/91f87a6443dbcc0260d6911856e054a62f8d3375) by cankanoa).
- Add benchmarking script and enhance global_match with parallelism ([604a1c0](https://github.com/spectralmatch/spectralmatch/commit/604a1c01a761a268f3e8ae41b464c4b8058d4c3d) by iosefa).
- Added check for raster requirements and removed get metadata function ([93ab20b](https://github.com/spectralmatch/spectralmatch/commit/93ab20bd72a8e8eb8abd4f7814f9e44b24efa8b8) by cankanoa).
- Add optional correction method to local_match function ([59a1668](https://github.com/spectralmatch/spectralmatch/commit/59a166882afa3f780e36590c2942d0ea7973fe50) by cankanoa).

### Fixed

- Fix image name extraction for multiple input images ([bbded6a](https://github.com/spectralmatch/spectralmatch/commit/bbded6a723693c4c8afe578626ddcbfe19c7bf33) by cankanoa).
- Fix incorrect working directory paths in example scripts ([f104ff1](https://github.com/spectralmatch/spectralmatch/commit/f104ff11a44b9f6bb8abb564b0b7539241fd2417) by cankanoa).
- Fix regex string formatting and remove unused import. ([ec25136](https://github.com/spectralmatch/spectralmatch/commit/ec25136f72801b4d2fae2c8b4d2d4a8531fb46d7) by cankanoa).
- Fix misaligned navigation item in mkdocs.yml ([2992a2b](https://github.com/spectralmatch/spectralmatch/commit/2992a2b98d190df6a87a1ea8fabb1ed1a4ea8d07) by cankanoa).
- Fix list formatting in docs ([067f9ad](https://github.com/spectralmatch/spectralmatch/commit/067f9ada6e1ba20dba4026e138663beaeef00e8e) by cankanoa).
- Fix overlap stats check and add comments in global_regression ([37660c1](https://github.com/spectralmatch/spectralmatch/commit/37660c194a801a3e47c0b283f0a2a992e2be3693) by cankanoa).
- Fixed bug where reference block means on the edges of images are calculated incorrectly. ([898d86c](https://github.com/spectralmatch/spectralmatch/commit/898d86c329130e010bb6faacaa09e98df774a9cc) by cankanoa).
- Fix format of docstrings to google format ([d634d37](https://github.com/spectralmatch/spectralmatch/commit/d634d37422bb9861c41acafcbc4e99e9a35d2527) by cankanoa).
- Fix imports in test_utils_common.py to match updated structure ([353f1f4](https://github.com/spectralmatch/spectralmatch/commit/353f1f467a2495b81decf3031a46ab756f12ece6) by cankanoa).

### Removed

- Remove unused imports across multiple modules ([ab1435e](https://github.com/spectralmatch/spectralmatch/commit/ab1435e93117025e626b7b4f54b766d12851ddd0) by cankanoa).
- Remove IntelliJ project configuration files ([0a83755](https://github.com/spectralmatch/spectralmatch/commit/0a83755abf4ff38182be3fb45ec6ed4b3fb47653) by cankanoa).
- Removed markov seamline creation ([601eb7f](https://github.com/spectralmatch/spectralmatch/commit/601eb7fe7b77e081695557b6eee49ddd1f9a011d) by cankanoa).
- Remove unused _compute_union_polygon function. ([f709c09](https://github.com/spectralmatch/spectralmatch/commit/f709c09a228883dba1f9c993753e0c596aa01fd0) by cankanoa).
- Removed import objgraph ([173c78a](https://github.com/spectralmatch/spectralmatch/commit/173c78aa49e82285a62c608d23bea9897ecedf74) by cankanoa).
- Remove custom.css and enhance debug output in local block adjustment ([30f30ad](https://github.com/spectralmatch/spectralmatch/commit/30f30ad68efaf1910a3a0231a909b2e8bbe3ac31) by cankanoa).

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

