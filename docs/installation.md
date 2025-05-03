# Installation Methods

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