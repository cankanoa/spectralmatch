import re
import rasterio
import os
import numpy as np

from omnicloudmask import predict_from_array
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from concurrent.futures import as_completed
from typing import Any
from typing import Tuple

from ..types_and_validation import Universal
from ..handlers import _resolve_paths, _resolve_output_dtype, _resolve_nodata_value
from ..utils_multiprocessing import (
    _resolve_parallel_config,
    _get_executor,
    WorkerContext,
    _resolve_windows,
)


def create_cloud_mask_with_omnicloudmask(
    input_images: Universal.SearchFolderOrListFiles,
    output_images: Universal.CreateInFolderOrListFiles,
    red_band_index: int,
    green_band_index: int,
    nir_band_index: int,
    *,
    down_sample_m: float = None,
    debug_logs: Universal.DebugLogs = False,
    image_parallel_workers: Universal.ImageParallelWorkers = None,
    omnicloud_kwargs: dict | None = None,
):
    """
    Generates cloud masks from input images using OmniCloudMask, with optional downsampling and multiprocessing.

    Args:
        input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_images (str | List[str], required): Defines output files from a template path, folder, or list of paths (with the same length as the input). Specify like: "/input/files/$.tif", "/input/folder" (assumes $_CloudMask.tif), ["/input/one.tif", "/input/two.tif"].
        red_band_index (int): Index of red band in the image.
        green_band_index (int): Index of green band in the image.
        nir_band_index (int): Index of NIR band in the image.
        down_sample_m (float, optional): If set, resamples input to this resolution in meters.
        debug_logs (bool, optional): If True, prints progress and debug info.
        image_parallel_workers (ImageParallelWorkers, optional): Enables parallel execution. Note: "process" does not work on macOS due to PyTorch MPS limitations.
        omnicloud_kwargs (dict | None): Additional arguments forwarded to predict_from_array.

    Raises:
        Exception: Propagates any error from processing individual images.
    """

    print("Start omnicloudmask")
    Universal.validate(
        input_images=input_images, output_images=output_images, debug_logs=debug_logs
    )

    input_image_paths = _resolve_paths(
        "search", input_images, kwargs={"default_file_pattern": "*.tif"}
    )
    output_image_paths = _resolve_paths(
        "create",
        output_images,
        kwargs={
            "paths_or_bases": input_image_paths,
            "default_file_pattern": "$_CloudClip.tif",
        },
    )
    image_parallel, image_backend, image_max_workers = _resolve_parallel_config(
        image_parallel_workers
    )

    if debug_logs:
        print(f"Input images: {input_image_paths}")
    if debug_logs:
        print(f"Output images: {output_image_paths}")

    image_args = [
        (
            input_path,
            output_path,
            red_band_index,
            green_band_index,
            nir_band_index,
            down_sample_m,
            debug_logs,
            omnicloud_kwargs,
        )
        for input_path, output_path in zip(input_image_paths, output_image_paths)
    ]

    if image_parallel:
        with _get_executor(image_backend, image_max_workers) as executor:
            futures = [
                executor.submit(_process_cloud_mask_image, *args) for args in image_args
            ]
            for future in as_completed(futures):
                future.result()
    else:
        for args in image_args:
            _process_cloud_mask_image(*args)


def _process_cloud_mask_image(
    input_image_path: str,
    output_mask_path: str,
    red_band_index: int,
    green_band_index: int,
    nir_band_index: int,
    down_sample_m: float,
    debug_logs: bool,
    omnicloud_kwargs: dict | None,
):
    """
    Processes a single image to generate a cloud mask using OmniCloudMask.

    Args:
        input_image_path (str): Path to input image.
        output_mask_path (str): Path to save output mask.
        red_band_index (int): Index of red band.
        green_band_index (int): Index of green band.
        nir_band_index (int): Index of NIR band.
        down_sample_m (float): Target resolution (if resampling).
        debug_logs (bool): If True, print progress info.
        omnicloud_kwargs (dict | None): Passed to predict_from_array.

    Raises:
        Exception: If any step in reading, prediction, or writing fails.
    """
    if omnicloud_kwargs is None:
        omnicloud_kwargs = {}

    with rasterio.open(input_image_path) as src:
        if down_sample_m is not None:
            left, bottom, right, top = src.bounds
            new_width = int((right - left) / down_sample_m)
            new_height = int((top - bottom) / down_sample_m)
            new_transform = from_origin(left, top, down_sample_m, down_sample_m)
            red = src.read(
                red_band_index,
                out_shape=(new_height, new_width),
                resampling=Resampling.bilinear,
            )
            green = src.read(
                green_band_index,
                out_shape=(new_height, new_width),
                resampling=Resampling.bilinear,
            )
            nir = src.read(
                nir_band_index,
                out_shape=(new_height, new_width),
                resampling=Resampling.bilinear,
            )
            meta = src.meta.copy()
            meta.update(
                {
                    "width": new_width,
                    "height": new_height,
                    "transform": new_transform,
                }
            )
        else:
            red = src.read(red_band_index)
            green = src.read(green_band_index)
            nir = src.read(nir_band_index)
            meta = src.meta.copy()

    band_array = np.stack([red, green, nir], axis=0)
    pred_mask = predict_from_array(band_array, **omnicloud_kwargs)
    pred_mask = np.squeeze(pred_mask)

    meta.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": pred_mask.dtype,
            "nodata": 0,
        }
    )

    with rasterio.open(output_mask_path, "w", **meta) as dst:
        dst.write(pred_mask, 1)


def create_ndvi_raster(
    input_images: Universal.SearchFolderOrListFiles,
    output_images: Universal.CreateInFolderOrListFiles,
    nir_band_index: int,
    red_band_index: int,
    *,
    custom_output_dtype: Universal.CustomOutputDtype = "float32",
    window_size: Universal.WindowSize = None,
    debug_logs: Universal.DebugLogs = False,
    image_parallel_workers: Universal.ImageParallelWorkers = None,
    window_parallel_workers: Universal.WindowParallelWorkers = None,
) -> None:
    """Computes NDVI masks for one or more images and writes them to disk.

    Args:
        input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_images (str | List[str], required): Defines output files from a template path, folder, or list of paths (with the same length as the input). Specify like: "/input/files/$.tif", "/input/folder" (assumes $_Vegetation.tif), ["/input/one.tif", "/input/two.tif"].
        nir_band_index: Band index for NIR (1-based).
        red_band_index: Band index for Red (1-based).
        custom_output_dtype: Optional output data type (e.g., "float32").
        window_size: Tile size or mode for window-based processing.
        debug_logs: Whether to print debug messages.
        image_parallel_workers: Parallelism strategy for image-level processing.
        window_parallel_workers: Parallelism strategy for window-level processing.

    Output:
        NDVI raster saved to output_images.
    """

    print("Start create NDVI rasters")
    Universal.validate(
        input_images=input_images,
        output_images=output_images,
        custom_output_dtype=custom_output_dtype,
        window_size=window_size,
        debug_logs=debug_logs,
        image_parallel_workers=image_parallel_workers,
        window_parallel_workers=window_parallel_workers,
    )

    input_image_paths = _resolve_paths(
        "search", input_images, kwargs={"default_file_pattern": "*.tif"}
    )
    output_image_paths = _resolve_paths(
        "create",
        output_images,
        kwargs={
            "paths_or_bases": input_image_paths,
            "default_file_pattern": "$_Vegetation.tif",
        },
    )
    image_names = _resolve_paths("name", input_image_paths)

    image_parallel, image_backend, image_max_workers = _resolve_parallel_config(
        image_parallel_workers
    )

    image_args = [
        (
            in_path,
            out_path,
            image_name,
            nir_band_index,
            red_band_index,
            custom_output_dtype,
            window_size,
            debug_logs,
            window_parallel_workers,
        )
        for in_path, out_path, image_name in zip(
            input_image_paths, output_image_paths, image_names
        )
    ]

    if image_parallel:
        with _get_executor(image_backend, image_max_workers) as executor:
            futures = [
                executor.submit(_ndvi_process_image, *args) for args in image_args
            ]
            for f in as_completed(futures):
                f.result()
    else:
        for args in image_args:
            _ndvi_process_image(*args)


def _ndvi_process_image(
    input_path: str,
    output_path: str,
    image_name: str,
    nir_band_index: int,
    red_band_index: int,
    custom_output_dtype: Universal.CustomOutputDtype,
    window_size: Universal.WindowSizeWithBlock,
    debug_logs: Universal.DebugLogs,
    window_parallel_workers: Universal.WindowParallelWorkers,
) -> None:
    """Processes a single image for NDVI using windowed strategy."""
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(dtype=_resolve_output_dtype(src, custom_output_dtype), count=1)

        with rasterio.open(output_path, "w", **profile) as dst:
            windows = _resolve_windows(src, window_size)
            window_args = [
                (image_name, window, nir_band_index, red_band_index, debug_logs)
                for window in windows
            ]

            window_parallel, backend, max_workers = _resolve_parallel_config(
                window_parallel_workers
            )
            if window_parallel:
                with _get_executor(
                    backend,
                    max_workers,
                    initializer=WorkerContext.init,
                    initargs=({image_name: ("raster", input_path)},),
                ) as executor:
                    futures = [
                        executor.submit(_ndvi_process_window, *args)
                        for args in window_args
                    ]
                    for f in as_completed(futures):
                        band, window, data = f.result()
                        dst.write(data, band, window)
            else:
                WorkerContext.init({image_name: ("raster", input_path)})
                for args in window_args:
                    band, window, data = _ndvi_process_window(*args)
                    dst.write(data, band, window)
                WorkerContext.close()


def _ndvi_process_window(
    image_name: str,
    window: rasterio.windows.Window,
    nir_band_index: int,
    red_band_index: int,
    debug_logs: bool,
) -> Tuple[int, rasterio.windows.Window, np.ndarray]:
    """Computes NDVI for a single window of a raster."""
    ds = WorkerContext.get(image_name)
    nir = ds.read(nir_band_index, window=window).astype(np.float32)
    red = ds.read(red_band_index, window=window).astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-9)

    return 1, window, ndvi


def band_math(
    input_images: Universal.SearchFolderOrListFiles,
    output_images: Universal.CreateInFolderOrListFiles,
    custom_math: str,
    *,
    debug_logs: Universal.DebugLogs = False,
    custom_nodata_value: Universal.CustomNodataValue = None,
    image_parallel_workers: Universal.ImageParallelWorkers = None,
    window_parallel_workers: Universal.WindowParallelWorkers = None,
    window_size: Universal.WindowSize = None,
    custom_output_dtype: Universal.CustomOutputDtype = None,
    calculation_dtype: Universal.CalculationDtype = None,
):
    """
    Applies custom band math expression to a list of input images and writes the results.

    Args:
        input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_images (str | List[str], required): Defines output files from a template path, folder, or list of paths (with the same length as the input). Specify like: "/input/files/$.tif", "/input/folder" (assumes $_Math.tif), ["/input/one.tif", "/input/two.tif"].
        custom_math (str): Python-compatible math expression using bands (e.g., "b1 + b2 / 2").
        debug_logs (bool, optional): If True, prints debug messages.
        custom_nodata_value (Any, optional): Override nodata value in source image.
        image_parallel_workers (int | str | None, optional): Controls image-level parallelism.
        window_parallel_workers (int | str | None, optional): Controls window-level parallelism.
        window_size (tuple[int, int] | None, optional): Size of processing windows (width, height).
        custom_output_dtype (str | None, optional): Output image data type (e.g., "uint16").
        calculation_dtype (str | None, optional): Computation data type (e.g., "float32").
    """

    input_image_paths = _resolve_paths(
        "search", input_images, kwargs={"default_file_pattern": "*.tif"}
    )
    output_image_paths = _resolve_paths(
        "create",
        output_images,
        kwargs={
            "paths_or_bases": input_image_paths,
            "default_file_pattern": "$_Math.tif",
        },
    )
    image_names = _resolve_paths("name", input_image_paths)

    with rasterio.open(input_image_paths[0]) as ds:
        nodata_value = _resolve_nodata_value(ds, custom_nodata_value)
        output_dtype = _resolve_output_dtype(ds, custom_output_dtype)

    image_parallel, image_backend, image_max_workers = _resolve_parallel_config(
        image_parallel_workers
    )

    # Extract referenced bands from custom_math (e.g., b1, b2, ...)
    band_indices = sorted(
        {int(match[1:]) for match in re.findall(r"\bb\d+\b", custom_math)}
    )

    image_args = [
        (
            in_path,
            out_path,
            name,
            custom_math,
            debug_logs,
            nodata_value,
            window_parallel_workers,
            window_size,
            band_indices,
            output_dtype,
            calculation_dtype,
        )
        for in_path, out_path, name in zip(
            input_image_paths, output_image_paths, image_names
        )
    ]

    if image_parallel:
        with _get_executor(image_backend, image_max_workers) as executor:
            futures = [
                executor.submit(_band_math_process_image, *arg) for arg in image_args
            ]
            for future in as_completed(futures):
                future.result()
    else:
        for arg in image_args:
            _band_math_process_image(*arg)


def _band_math_process_image(
    input_image_path: str,
    output_image_path: str,
    name: str,
    custom_math: str,
    debug_logs: bool,
    nodata_value,
    window_parallel_workers,
    window_size,
    band_indices,
    output_dtype,
    calculation_dtype,
):
    """
    Processes a single image by evaluating a custom math expression per pixel block.

    Args:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to save the result.
        name (str): Dataset identifier for use in worker context.
        custom_math (str): Math expression using band variables (e.g., "b1 - b2").
        debug_logs (bool): If True, prints debug information.
        nodata_value (Any): Value to treat as nodata during processing.
        window_parallel_workers (int | str | None): Parallelism setting for window processing.
        window_size (tuple[int, int]): Size of the processing window (width, height).
        band_indices (list[int]): List of 1-based band indices used in the expression.
        output_dtype (str): Output data type (e.g., "uint16").
        calculation_dtype (str): Intermediate computation data type (e.g., "float32").
    """

    with rasterio.open(input_image_path) as src:
        profile = src.profile.copy()
        profile.update(dtype=output_dtype, count=1, nodata=nodata_value)

        window_parallel, window_backend, window_max_workers = _resolve_parallel_config(
            window_parallel_workers
        )

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        with rasterio.open(output_image_path, "w", **profile) as dst:
            windows = _resolve_windows(src, window_size)
            args = [
                (
                    name,
                    window,
                    custom_math,
                    debug_logs,
                    nodata_value,
                    band_indices,
                    calculation_dtype,
                )
                for window in windows
            ]

            if window_parallel:
                with _get_executor(
                    window_backend,
                    window_max_workers,
                    initializer=WorkerContext.init,
                    initargs=({name: ("raster", input_image_path)},),
                ) as executor:
                    futures = [
                        executor.submit(_band_math_process_window, *arg) for arg in args
                    ]
                    for future in futures:
                        band, window, data = future.result()
                        dst.write(data.astype(output_dtype), band, window=window)
            else:
                WorkerContext.init({name: ("raster", input_image_path)})
                for arg in args:
                    band, window, data = _band_math_process_window(*arg)
                    dst.write(data.astype(output_dtype), band, window=window)
                WorkerContext.close()


def _band_math_process_window(
    name: str,
    window: rasterio.windows.Window,
    custom_math: str,
    debug_logs: bool,
    nodata_value,
    band_indices,
    calculation_dtype,
):
    """
    Computes the result of a band math expression within a raster window.

    Args:
        name (str): Dataset identifier to retrieve the open raster.
        window (rasterio.windows.Window): Raster window to process.
        custom_math (str): Math expression to evaluate (e.g., "b1 * b2").
        debug_logs (bool): If True, prints window-level debug messages.
        nodata_value (Any): Value representing nodata in the input bands.
        band_indices (list[int]): Band indices referenced in the expression.
        calculation_dtype (str): Data type used for evaluation.

    Returns:
        tuple: (band index, window, computed result as ndarray)
    """

    ds = WorkerContext.get(name)

    bands = [ds.read(i, window=window).astype(calculation_dtype) for i in band_indices]
    band_vars = {f"b{i}": b for i, b in zip(band_indices, bands)}

    try:
        result = eval(custom_math, {"np": np}, band_vars).astype(calculation_dtype)
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression '{custom_math}': {e}")

    if nodata_value is not None:
        nodata_mask = np.any([b == nodata_value for b in bands], axis=0)
        result[nodata_mask] = nodata_value

    return 1, window, result
