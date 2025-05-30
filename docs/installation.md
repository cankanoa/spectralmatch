# Installation Methods

---

## Installation as QGIS Plugin for Easy GUI Interface

### 1. [Download](https://qgis.org/download/) and install QGIS
### 2.	Open QGIS
### 3.	Go to Plugins → Manage and Install Plugins…
### 4.	Find spectralmatch in the list, install, and enable it
### 5.	Find the plugin in the Processing Toolbox

---

## Installation as a Python Library for use in Code (Recommended)

### 1. System requirements
Before installing, ensure you have the following system-level prerequisites:

- Python ≥ 3.10
- PROJ ≥ 9.3
- GDAL = 3.10.2
- pip

An easy way to install these dependancies is to use [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions):
```bash
conda create -n spectralmatch python=3.10 "gdal=3.10.2" "proj>=9.3" -c conda-forge
conda activate spectralmatch
```

### 2. Install spectralmatch

You can automatically install the library via [PyPI](https://pypi.org/). (this method installs only the core code as a library):

```bash
pip install spectralmatch
```

### 3. Run an example and modify for your use (optional)

Example scripts are provided to verify a successful installation and help you get started quickly in the repository at [`/docs/examples`](https://github.com/spectralmatch/spectralmatch/blob/main/docs/examples/) and downloadable via this [`link`](https://download-directory.github.io/?url=https://github.com/spectralmatch/spectralmatch/tree/main/docs/examples&filename=spectralmatch_examples).

---

## Installation as Python Code for Development and Customization

### 1. Clone the Repository
```bash
git clone https://github.com/spectralmatch/spectralmatch.git
cd spectralmatch
```

> Assuming you have Make installed, you can then run `make install-setup` to automatically complete the remaining setup steps.

### 2. System requirements
Before installing, ensure you have the following system-level prerequisites:

- Python ≥ 3.10
- PROJ ≥ 9.3
- GDAL = 3.10.2

An easy way to install these dependancies is to use [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions):
```bash
conda create -n spectralmatch python=3.10 "gdal=3.10.2" "proj>=9.3" -c conda-forge
conda activate spectralmatch
```

### 3. Install Dependancies (Optional Dev and Docs Dependancies)
The `pyproject.toml` defines **core** dependancies to run the library and optional **dev**, and **docs** dependancies.

```bash
pip install . # normal dependencies
pip install -e ".[dev]"   # developer dependencies
pip install -e ".[docs]"  # documentation dependencies
```

### 4. Read the [Contributing Guide](https://spectralmatch.github.io/spectralmatch/contributing/) if you aim to contribute