# spectralmatch: A toolkit for performing Relative Radiometric Normalization, with utilities for generating seamlines, cloud masks, Pseudo-Invariant Features, and statistics

[![Your-License-Badge](https://img.shields.io/badge/License-MIT-green)](#)
[![codecov](https://codecov.io/gh/cankanoa/spatialmatch/graph/badge.svg?token=OKAM0BUUNS)](https://codecov.io/gh/cankanoa/spatialmatch)
[![Open in Cloud Shell](https://img.shields.io/badge/Launch-Google_Cloud_Shell-blue?logo=googlecloud)](https://ssh.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https://github.com/spectralmatch/spectralmatch&cloudshell_working_dir=.&cloudshell_open_in_editor=docs/examples/example_global_to_local.py)

> [!IMPORTANT]
> This library is experimental and still under heavy development.
 
 ---

## Overview

![Global and Local Matching](./images/spectralmatch.png)

*spectralmatch* provides a Python library and QGIS plugin with multiple algorythms to perform Relative Radiometric Normalization (RRN). It also includes utilities for generating seamlines, cloud masks, Pseudo-Invariant Features, statistics, preprocessing, and more.

## Features

- **Automated:** Works without manual intervention, making it ideal for large-scale applications.

- **Consistent Multi-Image Analysis:** Ensures uniformity across images by applying systematic corrections with minimal spectral distortion.

- **Seamlessly Blended:** Creates smooth transitions between images without visible seams.

- **Unit Agnostic:** Works with any pixel unit and preserves the spectral information for accurate analysis. This inlcludes negative numbers and reflectance.

- **Better Input for Machine Learning Models:** Provides high-quality, unbiased data for AI and analytical workflows.

- **Minimizes Color Bias:** Avoids excessive color normalization and does not rely on a strict reference image.

- **Sensor Agnostic:** Works with all optical sensors. In addition, images from differnt sensors can be combined for multisensor analysis.

- **Parallel Processing:** Optimized for modern CPUs to handle large datasets efficiently.

- **Large-Scale Mosaics:** Designed to process and blend vast image collections effectively.
- **Time Series**: Normalize images across time with to compare spectral changes.

---

## Current Matching Algorithms

### Global to local matching
This technique is derived from 'An auto-adapting global-to-local color balancing method for optical imagery mosaic' by Yu et al., 2017 (DOI: 10.1016/j.isprsjprs.2017.08.002). It is particularly useful for very high-resolution imagery (satellite or otherwise) and works in a two phase process.
First, this method applies least squares regression to estimate scale and offset parameters that align the histograms of all images toward a shared spectral center. This is achieved by constructing a global model based on the overlapping areas of adjacent images, where the spectral relationships are defined. This global model ensures that each image conforms to a consistent radiometric baseline while preserving overall color fidelity.
However, global correction alone cannot capture intra-image variability so a second local adjustment phase is performed. The overlap areas are divided into smaller blocks, and each block’s mean is used to fine-tune the color correction. This block-wise tuning helps maintain local contrast and reduces visible seams, resulting in seamless and spectrally consistent mosaics with minimal distortion.


![Histogram matching graph](./images/matching_histogram.png)
*Shows the average spectral profile of two WorldView 3 images before and after global to local matching.*

#### Assumptions

- **Consistent Spectral Profile:** The true spectral response of overlapping areas remains the same throughout the images.

- **Least Squares Modeling:** A least squares approach can effectively model and fit all images' spectral profiles.

- **Scale and Offset Adjustment:** Applying scale and offset corrections can effectively harmonize images.

- **Minimized Color Differences:** The best color correction is achieved when color differences are minimized.

- **Geometric Alignment:** Images are assumed to be geometrically aligned with known relative positions.

- **Global Consistency:** Overlapping color differences are consistent across the entire image.

- **Local Adjustments:** Block-level color differences result from the global application of adjustments.

---

## Installation as a Python Library

### 1. System requirements
Before installing, ensure you have the following system-level prerequisites:

- Python ≥ 3.10
- PROJ ≥ 9.3
- GDAL ≥ 3.6

An easy way to install these dependancies is to use [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions):
```bash
conda create -n spectralmatch python>=3.10 gdal>=3.6 proj>=9.3 -c conda-forge
conda activate spectralmatch
```

### 2. Install spectralmatch (via PyPI or Source)

The recommended way to install is via [PyPI](https://pypi.org/). (this method installs only the core code as a library):

```bash
pip install spectralmatch
```


Another install method is to clone the repository and confugure the dependancies with `pyproject.toml`. (this method installs the whole repository for development or customization):

```bash
git clone https://github.com/spectralmatch/spectralmatch.git
cd spectralmatch
pip install .
```

### 3. Run example code and modify for use (optional)

Example scripts are provided to verify a successful installation and help you get started quickly at [`/docs/examples`](https://github.com/spectralmatch/spectralmatch/blob/main/docs/examples/)

---

## Installation as a QGIS Plugin

### 1. [Download](https://qgis.org/download/) and install QGIS
### 2.	Open QGIS
### 3.	Go to Plugins → Manage and Install Plugins…
### 4.	Find spectralmatch in the list, install, and enable it
### 5.	Find the plugin in the Processing Toolbox

---

## Documentation

Documentation is available at [spectralmatch.github.io/spectralmatch/](https://spectralmatch.github.io/spectralmatch/).

---

## Contributing Guide

We welcome all contributions the project! To get started:
1. [Create an issue](https://github.com/spectralmatch/spectralmatch/issues/new) with the appropriate label describing the feature or improvement. Provide relevant context, desired timeline, any assistance needed, who will be responsible for the work, anticipated results, and any other details.
2. [Fork the repository](https://github.com/spectralmatch/spectralmatch/fork) and create a new feature branch.
3. Make your changes and add any necessary tests.
4. Open a Pull Request against the main repository.

---

## Developer Guide

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/spectralmatch/spectralmatch.git
   cd spectralmatch
   ```

2. **Install with Dev andor Docs Extras**

   There are additional `[dev]` and `[docs]` dependancies specified in `pyproject.toml`:

   ```bash
   pip install -e ".[dev]"   # for developer dependencies
   pip install -e ".[docs]"  # for documentation dependencies
   ```

3. **Set Up Pre-commit Hooks**

   To maintain code consistency before each commit install these hooks:

   ```bash
   pre-commit install
   pre-commit run --all-files
   ```

---

## Testing

[pytest](https://docs.pytest.org/) is used for testing. Tests will automatically be run when merging into main but they can also be run locally via:

```bash
pytest
```

Run tests for a specific file or function:

```bash
pytest folder/file.py
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE.md) for details.