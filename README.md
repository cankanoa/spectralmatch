# spectralmatch: Global and Local Spectral Matching for Seamless Image Mosaics

[![Your-License-Badge](https://img.shields.io/badge/License-MIT-green)](#)

> [!IMPORTANT]
> This library is experimental and still under heavy development.
 
**Perform global and local histogram matching for multiple overlapping images to achieve seamless color consistency in mosaicked outputs.**

---

## Overview

*spectralmatch* uses least squares regression to balance colors across all images in a single global solution, then performs finer local adjustments on overlapping blocks. This two-phase process ensures high-quality color normalization with minimal spectral distortion.

![Global and Local Matching](./images/spectralmatch.png)

### Why Use spectralmatch?

- Fully automated color balancing of multiple overlapping images.  
- Minimizes color bias by avoiding a strict reference image.  
- Improves consistency with local, block-by-block refinements after global correction.  
- Easily integrated into new or existing remote sensing and GIS workflows.

---

## Features

- **Global Spectral Matching**  
  Calculates scale and offset across all images to reduce large color differences.

- **Local Spectral Matching**  
  Refines global corrections by applying fine-tuned color transformations in overlapping areas.

- **Sensor Agnostic**  
  Works on optical imagery from diverse sensors by assuming geometric alignment.

- **Parallel Processing**  
  Capable of handling large datasets efficiently on modern CPU cores.

- **Minimal Spectral Distortion**  
  Maintains the integrity of real color/spectral information in your data.

---

## Installation

### 1. System Requirements
Before installing *spectralmatch*, ensure you have the following system-level prerequisites:

- **Python ≥ 3.10**  
- **PROJ ≥ 9.3**  
- **GDAL ≥ 3.6** (verify using: `gdalinfo --version`)

### 2. Install spectralmatch (via PyPI or Source)

The recommended way to install *spectralmatch* is via [PyPI](https://pypi.org/):

```bash
pip install spectralmatch
```

*spectralmatch* includes a `pyproject.toml` which defines its Python dependencies. Installing via pip will automatically handle these. If you need to install from source, clone the repository and run:

```bash
pip install .
```

---

## Quick Start

After installation, you can use *spectralmatch* to perform global and local matching on multiple overlapping images:

```python
from spectralmatch.global_histogram_match import global_histogram_match
from spectralmatch.local_histogram_match import local_histogram_match

# Example: Basic usage

input_image_paths = [
    "image1.tif",
    "image2.tif",
    "image3.tif"
]

# 1. Global Matching
global_histogram_match(
    input_images=input_image_paths,
    output_folder="data/GlobalMatch",
    output_basename="_global",
    custom_mean_factor=3,   # adjust as needed
    custom_std_factor=1     # adjust as needed
)

# 2. Local Matching
local_histogram_match(
    input_images=[
        "data/GlobalMatch/images/image1_global.tif",
        "data/GlobalMatch/images/image2_global.tif",
        "data/GlobalMatch/images/image3_global.tif"
    ],
    output_folder="data/LocalMatch",
    output_basename="_local",
    target_blocks_per_image=100,
    projection="EPSG:XXXX", # specify your projection
    debug_mode=True         # optional debugging
)

print("Done! Check 'data/GlobalMatch' and 'data/LocalMatch' for results.")
```

Replace mentions of file paths, projection, and parameters as suitable for your data and environment.

---

## Documentation

Comprehensive documentation is forthcoming. In the meantime:  
- Refer to function docstrings for usage and parameter details.  
- Explore example scripts or tutorials within this repository for guidance.  
- Open an issue or discussion if you need further information.

---

## Developer Guides

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/spectralmatch.git
   ```
   Then navigate into the project folder:
   ```bash
   cd spectralmatch
   ```

2. **Install in Editable Mode with Dev Extras**  
   *spectralmatch* provides a `[dev]` extra in its `pyproject.toml` for development:

   ```bash
   pip install --upgrade pip
   pip install -e .[dev]   # for developer dependencies
   pip install -e .[docs]  # for documentation dependencies
   ```

3. **Set Up Pre-commit Hooks (Optional)**  	
   If you want to maintain consistency and code quality before each commit:

   ```bash
   pre-commit install
   pre-commit run --all-files
   ```

---

## Testing

*spectralmatch* uses [pytest](https://docs.pytest.org/) for testing. To run all tests:

```bash
pytest
```

Run tests for a specific file or function:

```bash
pytest tests/test_global_match.py
```

---

## Contributing

We welcome all contributions! To get started:  
1. Fork the repository and create a new feature branch.  
2. Make your changes and add any necessary tests.  
3. Open a Pull Request against the main repository.

We appreciate any feedback, suggestions, or pull requests to improve this project.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE.md) for details.