# Contributing Guide

Thank you for your interest in contributing. The sections below outline how the library is structured, how to submit changes, and the conventions to follow when developing new features or improving existing functionality.

For convenience, you can copy [this](/spectralmatch/llm_prompt/) auto updated LLM priming prompt with function headers and docs.

---

## Collaboration Instructions

We welcome all contributions the project! Please be respectful and work towards improving the library. To get started:

1. [Create an issue](https://github.com/spectralmatch/spectralmatch/issues/new) describing the feature or bug or just to ask a question. Provide relevant context, desired timeline, any assistance needed, who will be responsible for the work, anticipated results, and any other details.

2. [Fork the repository](https://github.com/spectralmatch/spectralmatch/fork) and create a new feature branch.

3. Make your changes and add any necessary tests.

4. Open a Pull Request against the main repository.

---

## Design Philosophy

 - Keep code concise and simple
 - Adapt code for large datasets with windows, multiprocessing, progressive computations, etc
 - Keep code modular and have descriptive names
 - Use PEP 8 code formatting
 - Use functions that are already created when possible
 - Combine similar params into one multi-value parameter
 - Use similar naming convention and input parameter format as other functions.
 - Create docstrings (Google style), tests, and update the docs for new functionality

---

## Extensible Function Types

In Relative Radiometric Normalization (RRN) methods often differ in how images are matched, pixels are selected, and seamlines are created. This library organizes those into distinct Python packages, while other operations like aligning rasters, applying masks, merging images, and calculating statistics are more consistent across techniques and are treated as standard utilities.

### Matching functions

Used to adjust the pixel values of images to ensure radiometric consistency across scenes. These functions compute differences between images and apply transformations so that brightness, contrast, or spectral characteristics align across datasets.


### Masking functions (PIF/RCS)

Used to define which parts of an image should be kept or discarded based on spatial criteria. These functions apply vector-based filters or logical rules to isolate regions of interest, remove clouds, or exclude invalid data from further processing.


### Seamline functions

Used to determine optimal boundaries between overlapping image regions. These functions generate cutlines that split image footprints in a way that minimizes visible seams and balances spatial coverage, often relying on geometric relationships between overlapping areas.

---

## Standard UI

Reusable types are organized into the types and validation module. Use these types directly as the types of params inside functions where applicable. Use the appropriate _resolve... function to resolve these inputs into usable variables.

### Input/Output
The input_name parameter defines how the input files are determined and accepts either a str or a list. If given as a str, it should contain either a folder glob pattern path and default_file_pattern must be set or a whole glob pattern file path. Functions should default to searching for all appropriately formated files in the input folder (for example "*.tif"). Alternatively, it can be a list of full file paths to individual input images. For example:

- input_images="/input/files/*.tif" (does not require default_file_pattern)
- input_images="/input/folder" (requires default_file_pattern to be set), 
- input_images=["/input/one.tif", "/input/two.tif", ...] (does not require default_file_pattern)

 The output_name parameter defines how output filenames are determined and accepts either a str or a list. If given as a str, it should contain either a folder template pattern path and default_file_pattern must be set or a whole template pattern file path. Functions should default to templating with basename, underscore, processing step (for example "$_Global"). Alternatively, it may be a list of full output paths, which must match the number of input images. For example:
 
- output_images="/output/files/$.tif" (does not require default_file_pattern)
- output_images="/output/folder" (requires default_file_pattern to be set), 
- output_images=["/output/one.tif", "/output/two.tif", ...] (does not require default_file_pattern)

The _resolve_paths function handles creating folders for output files. Folders and files are distinguished by the presence of a "." in the basename.
```python
# Params
input_name # For example: input_images
input_name # For example: output_images

# Types
SearchFolderOrListFiles = str | List[str] # Required
CreateInFolderOrListFiles = str | List[str] # Required

# Resolve
input_image_paths = _resolve_paths("search", input_images, kwargs={"default_file_pattern":"*.tif"})
output_image_paths = _resolve_paths("create", output_images, kwargs={"paths_or_bases":input_image_paths, "default_file_pattern":"$_Global.tif"})
image_names = _resolve_paths("name", input_image_paths)
# This pattern can also be used with other input types like vectors
```

### Output dtype
The custom_output_dtype parameter specifies the data type for output rasters and defaults to the input image’s data type if not provided.
```python
# Param
custom_output_dtype

# Type
CustomOutputDtype = str | None # Default: None

# Resolve
output_dtype = _resolve_output_dtype(rasterio.DatasetReader, custom_output_dtype)
```


### Nodata Value
The custom_nodata_value parameter overrides the input nodata value from the first raster in the input rasters if set. 
```python
# Param
custom_nodata_value

# Type
CustomNodataValue = float | int | None # Default: None

# Resolve
nodata_value = _resolve_nodata_value(rasterio.DatasetReader, custom_nodata_value)
```

### Debug Logs
The debug_logs parameter enables printing of debug information; it defaults to False. Functions should begin by printing "Start {process name}", while all other print statements should be conditional on debug_logs being True. When printing the image being processed, use the image name and not the image path.
```python
# Param
debug_logs

# Type
DebugLogs = bool # Default: False

# No resolve function necessary
```

### Vector Mask
The vector_mask parameter limits statistics calculations to specific areas and is given as a tuple with two or three items: a literal "include" or "exclude" to define how the mask is applied, a string path to the vector file, and an optional field name used to match geometries based on the input image name (substring match allowed). Defaults to None for no mask.

```python
# Param
vector_mask

# Type
VectorMask = Tuple[Literal["include", "exclude"], str, Optional[str]] | None

# No resolve function necessary
```

### Parallel Workers
The image_parallel_workers parameter defines the parallelization strategy at the image level. It accepts a tuple such as ("process", "cpu") to enable multiprocessing across all available CPU cores, or you can use "thread" as the backend if threading is preferred. Set it to None to disable image-level parallelism. The window_parallel_workers parameter controls parallelization within each image at the window level and follows the same format. Setting it to None disables window-level parallelism. Processing windows should be done one band at a time for scalability.
```python
# Params
image_parallel_workers
window_parallel_workers

# Types
ImageParallelWorkers = Tuple[Literal["process", "thread"], Literal["cpu"] | int] | None
WindowParallelWorkers = Tuple[Literal["process"], Literal["cpu"] | int] | None

# Resolve
image_parallel, image_backend, image_max_workers = _resolve_parallel_config(image_parallel_workers)
window_parallel, window_backend, window_max_workers = _resolve_parallel_config(window_parallel_workers)


# Main process example
image_args = [(arg, other_args, ...) for arg in inputs]
if image_parallel:
    with _get_executor(image_backend, image_max_workers) as executor:
        futures = [executor.submit(_name_process_image, *arg) for arg in image_args]
        for future in as_completed(futures):
                result = future.result()
else:
        for arg in image_args:
            result = _name_process_image(*arg)

def _name_process_image(image_name, arg_1, arg_2, ...):
    with rasterio.open(input_image_path) as src:
        # Open output image as well if saving to image
        windows = _resolve_windows(src, window_size)
        window_args = [(window, other_args, ...) for window in windows]

        with _get_executor(
            window_backend, 
            window_max_workers,
            initializer=WorkerContext.init,
            initargs=({image_name: ("raster", input_image_path)},)
            ) as executor:
            futures = [executor.submit(_name_process_window, *arg) for arg in window_args]
            for future in as_completed(futures):
                band, window, result = future.result()
                # Save result to variable or dataset
        else:
            WorkerContext.init({image_name: ("raster", input_image_path)})
            for arg in window_args:
                band, window, buf = _name_process_window(*arg)
                # Save result to variable or dataset
            WorkerContext.close()

def _name_process_window(image_name, arg_1, arg_2, ...):
    ds = WorkerContext.get(image_name)
    # Process result to return
    
    return band, window, data
```

### Windows
The window_size parameter sets the tile size for reading and writing, using an integer for square tiles, a tuple for custom dimensions, "internal" to use the raster’s native tiling (ideal for efficient streaming from COGs), or None to process the full image at once.
```python
# Param
window_size

# Types
WindowSize = int | Tuple[int, int] | Literal["internal"] | None
WindowSizeWithBlock = int | Tuple[int, int] | Literal["internal", "block"] | None

# Resolve
windows = _resolve_windows(rasterio.DatasetReader, window_size)
```

### COGs
The save_as_cog parameter, when set to True, saves the output as a Cloud-Optimized GeoTIFF with correct band and block ordering.
```python
# Param
SaveAsCog = bool # Default: True

# Type
SaveAsCog = bool # Default: True

# No resolve function necessary
```

---

## Validate Inputs
The validate methods are used to check that input parameters follow expected formats before processing begins. There are different validation methods for different scopes—some are general-purpose (e.g., Universal.validate) and others apply to specific contexts like matching (Match.validate_match). These functions raise clear errors when inputs are misconfigured, helping catch issues early and enforce consistent usage patterns across the library.
```python
# Validate params example
Universal.validate(
    input_images=input_images,
    output_images=output_images,
    vector_mask=vector_mask,
)
Match.validate_match(
    specify_model_images=specify_model_images,
    )
```

---

## File Cleanup
Temporary generated files can be deleted once they are no longer needed via this command:
```bash
make clean
```

---

## Docs
Docs are deployed on push or merge at the main branch, or use the following commands:

### Serve docs locally
Runs a local dev server at http://localhost:8000.
```bash
make docs-serve
```

### Build static site
Generates the static site into the site/ folder.

```bash
make docs-build
```

### Deploy to GitHub Pages
Deploys built site using mkdocs gh-deploy.
```bash
make docs-deploy
```
---

## Versioning
Automatically create a GitHub release, Pypi library, and QGIS plugin with each version. All three distributions are on the same versioning and deployed with GitHub actions. New versions will be released when sufficient new functionality or bug fixes have been added. 
```bash
make version version=1.2.3
```

---

## Code Formatting
This project uses [black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) for code formatting and ruff for linting.

### Set Up Pre-commit Hooks (Recommended)
To maintain code consistency use this hook to check and correct code formatting automatically:

```bash
pre-commit install
pre-commit run --all-files
```

### Manual Formatting

**Format code:** Automatically formats all Python files with black.

```bash
make format
```

**Check formatting:** Checks that all code is formatted (non-zero exit code if not).
```bash
make check-format
```

**Lint code:** Runs ruff to catch style and quality issues.
```bash
make lint
```

---

## Testing
[pytest](https://docs.pytest.org/) is used for testing. Tests will automatically be run when merging into main but they can also be run locally via:
```bash
make test
```

To test a individual folder or file:
```bash
make test-file path=path/to/folder_or_file
```

## Building Python Library and QGIS Plugin Locally
Use these commands to build packages locally:
```shell
make qgis-build # Build QGIS plugin
make python-build # Build python library
```