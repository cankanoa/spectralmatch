# spectralmatch: Global and Local Spectral Matching for Seamless Image Mosaics

[![Your-License-Badge](https://img.shields.io/badge/License-MIT-green)](#)

> [!IMPORTANT]
> This library is experimental and still under heavy development.
 
**Perform global and local histogram matching for multiple overlapping images to achieve seamless color consistency in mosaicked outputs.**

---

## Overview

*spectralmatch* uses least squares regression to balance colors across all images in a single global solution, then performs finer local adjustments on overlapping blocks. This two-phase process prduces high-quality color normalization with minimal spectral distortion. This technique is derived from 'An auto-adapting global-to-local color balancing method for optical imagery mosaic' by Yu et al., 2017 (DOI: 10.1016/j.isprsjprs.2017.08.002).

![Global and Local Matching](./images/spectralmatch.png)

---

## Features

- **Fully Automated:** Works without manual intervention, making it ideal for large-scale applications.

- **Consistent Multi-Image Analysis:** Ensures uniformity across images by applying systematic corrections with minimal spectral distortion.

- **Seamlessly Blended:** Creates smooth transitions between images without visible seams.

- **Unit Agnostic:** Works with any pixel unit and preserves the spectral information for accurate analysis. This inlcludes negative numbers and reflectance.

- **Better Input for Machine Learning Models:** Provides high-quality, unbiased data for AI and analytical workflows.

- **Minimizes Color Bias:** Avoids excessive color normalization and does not rely on a strict reference image.

- **Global Spectral Matching:** Computes scale and offset across all images to correct large spectral differences.

- **Local Spectral Matching:** Applies fine-tuned, block-by-block adjustments after global corrections.

- **Sensor Agnostic:** Works with all optical sensors. In addition, images from differnt sensors can be combined for multisensor analysis.

- **Parallel Processing:** Optimized for modern CPUs to handle large datasets efficiently.

- **Script Automation & Integration:** Easily incorporated into new or existing remote sensing and GIS workflows.

- **Large-Scale Mosaics:** Designed to process and blend vast image collections effectively.

## Assumptions

- **Consistent Spectral Profile:** The true spectral response of overlapping areas remains the same throughout the images.

- **Least Squares Modeling:** A least squares approach can effectively model and fit all images' spectral profiles.

- **Scale and Offset Adjustment:** Applying scale and offset corrections can effectively harmonize images.

- **Minimized Color Differences:** The best color correction is achieved when color differences are minimized.

- **Geometric Alignment:** Images are assumed to be geometrically aligned with known relative positions.

- **Global Consistency:** Overlapping color differences are consistent across the entire image.

- **Local Adjustments:** Block-level color differences result from the global application of adjustments.

## Whats happening to the images?
The color balancing process shifts the histograms of the images toward a common center, ensuring spectral consistency across the dataset. Each image has its own unique scale and offset applied to bring it closer to this central distribution. This is achieved by constructing a global model based on the overlapping areas of adjacent images, where the spectral relationships are defined. The global correction adjusts each image’s scale and offset so that their histograms align with the central tendency of all images.

However, a global correction alone, based on a single mean, does not fully account for variations within individual images. To refine the adjustment locally, the overlap areas are divided into smaller blocks, and each block’s mean is used to fine-tune the color correction. This ensures that the local differences within images are better preserved, leading to seamless and natural-looking colors.

![Histogram matching graph](./images/matching_histogram.png)

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
import os

from spectralmatch.process import global_match, local_match
script_dir = os.path.dirname(os.path.abspath(__file__))

# -------------------- Global params
input_folder = os.path.join(script_dir, "input")
global_folder = os.path.join(script_dir, "output/global_match")  # This is the output of global match
custom_mean_factor = 3  # Defualt 1; 3 often works better to 'move' the spectral mean of images closer together
custom_std_factor = 1  # Defualt 1

# -------------------- Local params
local_folder = os.path.join(script_dir, "output/local_match")


# -------------------- Global Histogram Match Mulispectral Images
input_image_paths_array = [
    os.path.join(input_folder, f)
    for f in os.listdir(input_folder)
    if f.lower().endswith(".tif")
]

global_match(
    input_image_paths_array,
    global_folder,
    custom_mean_factor,
    custom_std_factor,
)

# -------------------- Local Histogram Match Mulispectral Images
global_image_paths_array = [
    os.path.join(f"{global_folder}/images", f)
    for f in os.listdir(f"{global_folder}/images")
    if f.lower().endswith(".tif")
]

local_match(
    global_image_paths_array,
    local_folder,
    target_blocks_per_image=100,
    projection="EPSG:6635",
    debug_mode=True,
    global_nodata_value=-32768,
)

print("Done with global and local histogram matching")
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
   pip install -e ".[dev]"   # for developer dependencies
   pip install -e ".[docs]"  # for documentation dependencies
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