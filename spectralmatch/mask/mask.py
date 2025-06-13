import re
import rasterio
import os
import numpy as np
import geopandas as gpd

from omnicloudmask import predict_from_array
from rasterio.features import shapes
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from concurrent.futures import as_completed
from typing import Any
from shapely.geometry import shape
from typing import Tuple

from ..types_and_validation import Universal
from ..handlers import _resolve_paths, _resolve_output_dtype, _resolve_nodata_value
from ..utils_multiprocessing import _resolve_parallel_config, _get_executor, WorkerContext, _resolve_windows


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
    **omnicloud_kwargs: Any,
    ):
    """
    Generates cloud masks from input images using OmniCloudMask, with optional downsampling and multiprocessing.

    Args:
        input_images (SearchFolderOrListFiles): Input folder and pattern or list of image paths.
        output_images (CreateInFolderOrListFiles): Output folder and pattern or list of paths.
        red_band_index (int): Index of red band in the image.
        green_band_index (int): Index of green band in the image.
        nir_band_index (int): Index of NIR band in the image.
        down_sample_m (float, optional): If set, resamples input to this resolution in meters.
        debug_logs (bool, optional): If True, prints progress and debug info.
        image_parallel_workers (ImageParallelWorkers, optional): Enables parallel execution. Note: "process" does not work on macOS due to PyTorch MPS limitations.
        **omnicloud_kwargs: Additional arguments forwarded to predict_from_array.

    Raises:
        Exception: Propagates any error from processing individual images.
    """

    print("Start omnicloudmask")
    Universal.validate(
        input_images=input_images,
        output_images=output_images,
        debug_logs=debug_logs
    )

    input_image_paths = _resolve_paths("search", input_images)
    output_image_paths = _resolve_paths("create", output_images, (input_image_paths,))
    image_parallel, image_backend, image_max_workers = _resolve_parallel_config(image_parallel_workers)

    if debug_logs: print(f"Input images: {input_image_paths}")
    if debug_logs: print(f"Output images: {output_image_paths}")

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
            futures = [executor.submit(_process_cloud_mask_image, *args) for args in image_args]
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
    omnicloud_kwargs: dict,
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
        omnicloud_kwargs (dict): Passed to predict_from_array.

    Raises:
        Exception: If any step in reading, prediction, or writing fails.
    """

    with rasterio.open(input_image_path) as src:
        if down_sample_m is not None:
            left, bottom, right, top = src.bounds
            new_width = int((right - left) / down_sample_m)
            new_height = int((top - bottom) / down_sample_m)
            new_transform = from_origin(left, top, down_sample_m, down_sample_m)
            red = src.read(red_band_index, out_shape=(new_height, new_width), resampling=Resampling.bilinear)
            green = src.read(green_band_index, out_shape=(new_height, new_width), resampling=Resampling.bilinear)
            nir = src.read(nir_band_index, out_shape=(new_height, new_width), resampling=Resampling.bilinear)
            meta = src.meta.copy()
            meta.update({
                'width': new_width,
                'height': new_height,
                'transform': new_transform,
            })
        else:
            red = src.read(red_band_index)
            green = src.read(green_band_index)
            nir = src.read(nir_band_index)
            meta = src.meta.copy()

    band_array = np.stack([red, green, nir], axis=0)
    pred_mask = predict_from_array(band_array, **omnicloud_kwargs)
    pred_mask = np.squeeze(pred_mask)

    meta.update({
        'driver': 'GTiff',
        'count': 1,
        'dtype': pred_mask.dtype,
        'nodata': 0,
    })

    with rasterio.open(output_mask_path, 'w', **meta) as dst:
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
        input_images: Folder and pattern tuple or list of input image paths.
        output_images: Output folder and template or list of output paths.
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

    input_paths = _resolve_paths("search", input_images)
    output_paths = _resolve_paths("create", output_images, (input_paths,))
    image_names = _resolve_paths("name", input_paths)

    image_parallel, image_backend, image_max_workers = _resolve_parallel_config(image_parallel_workers)

    image_args = [
        (in_path, out_path, image_name, nir_band_index, red_band_index, custom_output_dtype, window_size, debug_logs, window_parallel_workers)
        for in_path, out_path, image_name in zip(input_paths, output_paths, image_names)
    ]

    if image_parallel:
        with _get_executor(image_backend, image_max_workers) as executor:
            futures = [executor.submit(_ndvi_process_image, *args) for args in image_args]
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

        with rasterio.open(output_path, 'w', **profile) as dst:
            windows = _resolve_windows(src, window_size)
            window_args = [(image_name, window, nir_band_index, red_band_index, debug_logs) for window in windows]

            window_parallel, backend, max_workers = _resolve_parallel_config(window_parallel_workers)
            if window_parallel:
                with _get_executor(backend, max_workers,
                                   initializer=WorkerContext.init,
                                   initargs=({image_name: ("raster", input_path)},)) as executor:
                    futures = [executor.submit(_ndvi_process_window, *args) for args in window_args]
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


def process_raster_values_to_vector_polygons(
    input_images: Universal.SearchFolderOrListFiles,
    output_vectors: Universal.CreateInFolderOrListFiles,
    *,
    custom_nodata_value: Universal.CustomNodataValue = None,
    custom_output_dtype: Universal.CustomOutputDtype = None,
    image_parallel_workers: Universal.ImageParallelWorkers = None,
    window_parallel_workers: Universal.WindowParallelWorkers = None,
    window_size: Universal.WindowSizeWithBlock = None,
    debug_logs: Universal.DebugLogs = False,
    extraction_expression: str,
    filter_by_polygon_size: str = None,
    polygon_buffer: float = 0.0,
    value_mapping: dict = None,
    ):
    """
    Converts raster values into vector polygons based on an expression and optional filtering logic.

    Args:
        input_images (Universal.SearchFolderOrListFiles): Either a (folder, pattern) tuple to search for rasters or a list of raster file paths.
        output_vectors (Universal.CreateInFolderOrListFiles): Output folder or list of file paths where the vector polygons will be saved.
        custom_nodata_value (Universal.CustomNodataValue, optional): Custom NoData value to override the default from the raster metadata.
        custom_output_dtype (Universal.CustomOutputDtype, optional): Desired output data type. If not set, defaults to raster’s dtype.
        image_parallel_workers (Universal.ImageParallelWorkers, optional): Controls parallelism across input images. Can be an integer, executor string, or boolean.
        window_parallel_workers (Universal.WindowParallelWorkers, optional): Controls parallelism within a single image by processing windows in parallel.
        window_size (Universal.WindowSizeWithBlock, optional): Size of each processing block (width, height), or a strategy string such as "block" or "whole".
        debug_logs (Universal.DebugLogs, optional): Whether to print debug logs to the console.
        extraction_expression (str): Logical expression to identify pixels of interest using band references (e.g., "b1 > 10 & b2 < 50").
        filter_by_polygon_size (str, optional): Area filter for resulting polygons. Can be a number (e.g., ">100") or percentile (e.g., "95%").
        polygon_buffer (float, optional): Distance in coordinate units to buffer the resulting polygons. Default is 0.
        value_mapping (dict, optional): Mapping from original raster values to new values. Use `None` to convert to NoData.

    """

    print("Start raster value extraction to polygons")

    Universal.validate(
        input_images=input_images,
        output_images=output_vectors,
        custom_nodata_value=custom_nodata_value,
        custom_output_dtype=custom_output_dtype,
        image_parallel_workers=image_parallel_workers,
        window_parallel_workers=window_parallel_workers,
        window_size=window_size,
        debug_logs=debug_logs,

    )

    input_image_paths = _resolve_paths("search", input_images)
    output_image_paths = _resolve_paths("create", output_vectors, (input_image_paths,))

    image_parallel, image_backend, image_max_workers = _resolve_parallel_config(image_parallel_workers)
    window_parallel, window_backend, window_max_workers = _resolve_parallel_config(window_parallel_workers)

    image_args = [
        (in_path, out_path, extraction_expression, filter_by_polygon_size, polygon_buffer, value_mapping, custom_nodata_value, custom_output_dtype, window_parallel, window_backend, window_max_workers, window_size, debug_logs)
        for in_path, out_path in zip(input_image_paths, output_image_paths)
    ]

    if image_parallel:
        with _get_executor(image_backend, image_max_workers) as executor:
            futures = [executor.submit(_process_image_to_polygons, *args) for args in image_args]
            for future in as_completed(futures):
                future.result()
    else:
        for args in image_args:
            _process_image_to_polygons(*args)


def _process_image_to_polygons(
    input_image_path,
    output_vector_path,
    extraction_expression,
    filter_by_polygon_size,
    polygon_buffer,
    value_mapping,
    custom_nodata_value,
    custom_output_dtype,
    window_parallel,
    window_backend,
    window_max_workers,
    window_size,
    debug_logs,
    ):
    """
    Processes a single raster file and extracts polygons based on logical expressions and optional filters.

    Args:
        input_image_path (str): Path to the input raster image.
        output_vector_path (str): Output file path for the resulting vector file (GeoPackage format).
        extraction_expression (str): Logical expression using band indices (e.g., "b1 > 5 & b2 < 10").
        filter_by_polygon_size (str): Area filter for polygons. Supports direct comparisons (">100") or percentiles ("90%").
        polygon_buffer (float): Amount of buffer to apply to polygons in projection units.
        value_mapping (dict): Dictionary mapping original raster values to new ones. Set value to `None` to mark as NoData.
        custom_nodata_value: Custom NoData value to use during processing.
        custom_output_dtype: Output data type for raster if relevant in future I/O steps.
        window_parallel: Whether to parallelize over raster windows.
        window_backend: Backend used for window-level parallelism (e.g., "thread", "process").
        window_max_workers: Max number of parallel workers for window-level processing.
        window_size: Tuple or strategy defining how the raster should be split into windows.
        debug_logs (bool): Whether to print debug logging information.
    """

    if debug_logs:
        print(f"Processing {input_image_path}")

    with rasterio.open(input_image_path) as src:
        crs = src.crs
        nodata_value = _resolve_nodata_value(src, custom_nodata_value)
        dtype = _resolve_output_dtype(src, custom_output_dtype)

        band_indices = sorted(set(int(b[1:]) for b in re.findall(r"b\d+", extraction_expression)))
        band_indices = sorted(set(band_indices))

        windows = _resolve_windows(src, window_size)
        window_args = [(w, band_indices, extraction_expression, value_mapping, nodata_value) for w in windows]

        polygons = []
        if window_parallel:
            with _get_executor(
                window_backend,
                window_max_workers,
                initializer=WorkerContext.init,
                initargs=({"input": ("raster", input_image_path)},),
            ) as executor:
                futures = [executor.submit(_process_window, *args) for args in window_args]
                for f in as_completed(futures):
                    polygons.extend(f.result())
        else:
            WorkerContext.init({"input": ("raster", input_image_path)})
            for args in window_args:
                polygons.extend(_process_window(*args))
            WorkerContext.close()

    if not polygons:
        if debug_logs: print("No features found.")
        return

    gdf = gpd.GeoDataFrame(polygons, crs=crs)
    merged = gdf.dissolve(by="value", as_index=False)

    if filter_by_polygon_size:
        if filter_by_polygon_size.endswith("%"):
            pct = float(filter_by_polygon_size.strip("%"))
            area_thresh = np.percentile(merged.geometry.area, pct)
        else:
            op, val = filter_by_polygon_size[:2], filter_by_polygon_size[2:]
            if not op[1] in "=<>":
                op, val = filter_by_polygon_size[:1], filter_by_polygon_size[1:]
            area_thresh = float(val)

        op_map = {
            "<": lambda x: x < area_thresh,
            "<=" : lambda x: x <= area_thresh,
            ">": lambda x: x > area_thresh,
            ">=" : lambda x: x >= area_thresh,
            "==" : lambda x: x == area_thresh,
            "!=" : lambda x: x != area_thresh,
        }
        merged = merged[op_map[op](merged.geometry.area)]

    if polygon_buffer:
        merged["geometry"] = merged.geometry.buffer(polygon_buffer)

    if os.path.exists(output_vector_path):
        os.remove(output_vector_path)

    merged.to_file(output_vector_path, driver="GPKG", layer="mask")


def _process_window(
    window,
    band_indices,
    expression,
    value_mapping,
    nodata_value
    ):
    """
    Processes a single window of a raster image to extract polygons matching an expression.

    Args:
        window (rasterio.windows.Window): Raster window to process.
        band_indices (list[int]): List of band indices required by the expression (e.g., [1, 2]).
        expression (str): Logical expression involving bands (e.g., "b1 > 10 & b2 < 50").
        value_mapping (dict): Dictionary mapping original raster values to new ones or to NoData.
        nodata_value (int | float): NoData value to exclude from analysis.

    Returns:
        list[dict]: List of dictionaries with keys `"value"` and `"geometry"` representing polygons.
    """

    src = WorkerContext.get("input")
    bands = [src.read(i, window=window) for i in band_indices]
    data = np.stack(bands, axis=0)

    if value_mapping:
        for orig, val in value_mapping.items():
            if val is None:
                data[data == orig] = nodata_value
            else:
                data[data == orig] = val

    pattern = re.compile(r"b(\d+)")
    expr = pattern.sub(lambda m: f"data[{int(m.group(1)) - 1}]", expression)
    mask = eval(expr).astype(np.uint8)

    results = []
    for s, v in shapes(mask, mask=mask, transform=src.window_transform(window)):
        if v != 1:
            continue
        geom = shape(s)
        results.append({"value": 1, "geometry": geom})

    return results