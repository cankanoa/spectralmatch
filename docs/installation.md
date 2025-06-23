# Installation Methods

---

## Installation as a QGIS Plugin
### 1. Get QGIS
[Download](https://qgis.org/download/) and install QGIS.
> This plugin requires python>=3.10. QGIS ships with different versions of Python, to check, in the QGIS menu, go to QGIS>About gis. If your version is out of date you can use `conda install qgis` to create a containerized version of QGIS and then `qgis` to start the program.
### 2. Install spactalmatch plugin
- Go to Plugins → Manage and Install Plugins…
- Find spectralmatch in the list, install, and enable it
- Find the plugin in the Processing Toolbox

---

## Installation as a Python Library and CLI

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

---

## Installation from Source

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

### 3. Install Dependancies
The `pyproject.toml` defines **core** dependancies to run the library and optional **dev**, and **docs** dependancies.

```bash
pip install . # normal dependencies
pip install -e ".[dev]"   # developer dependencies
pip install -e ".[docs]"  # documentation dependencies
```

### 4. Read the [Contributing Guide](https://spectralmatch.github.io/spectralmatch/contributing/) if you aim to contribute