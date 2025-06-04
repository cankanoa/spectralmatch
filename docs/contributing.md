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
The input_images parameter accepts either a tuple or a list. If given as a tuple, it should contain a folder path and a glob pattern to search for files (e.g., ("/input/folder", "*.tif")). Alternatively, it can be a list of full file paths to individual input images. The output_images parameter defines how output filenames are determined. It can also be a tuple, consisting of an output folder and a filename template where "\$" is replaced with each input image’s basename (e.g., ("/output/folder", "$_GlobalMatch.tif")). Alternatively, it may be a list of full output paths, which must match the number of input images.
```python
# Params
input_images
output_images

# Types
SearchFolderOrListFiles = Tuple[str, str] | List[str] # Required
CreateInFolderOrListFiles = Tuple[str, str] | List[str] # Required

# Resolve
input_image_paths = _resolve_paths("search", input_images)
output_image_paths = _resolve_paths("create", output_images, (input_image_paths,))
```

### Nodata Value
The output_dtype parameter specifies the data type for output rasters and defaults to the input image’s data type if not provided or None. Functions should begin by printing "Start {process name}", while all other print statements should be conditional on debug_logs being True.
```python
# Param
custom_nodata_value

# Type
CustomNodataValue = float | int | None # Default: None

# No resolve function necessary
```

### Debug Logs
The debug_logs parameter enables printing of debug information and constraint matrices when set to True; it defaults to False.
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
    
    return band, window, block
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
Uses git tag to create annotated version tags and push them. This also syncs to Pypi. New versions will be released when the maintainer determines sufficient new functionality has been added.
```bash
make tag version=1.2.3
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