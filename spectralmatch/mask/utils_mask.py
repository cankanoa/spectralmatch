import os
import rasterio
import numpy as np
import re
import geopandas as gpd

from shapely.geometry import shape
from typing import Tuple
from rasterio.features import shapes
from concurrent.futures import as_completed

from ..utils_multiprocessing import (
    _get_executor,
    WorkerContext,
    _resolve_windows,
    _resolve_parallel_config,
)
from ..handlers import _resolve_paths, _resolve_nodata_value, _resolve_output_dtype
from ..types_and_validation import Universal


def threshold_raster(
    input_images: Universal.SearchFolderOrListFiles,
    output_images: Universal.CreateInFolderOrListFiles,
    threshold_math: str,
    *,
    debug_logs: Universal.DebugLogs = False,
    custom_nodata_value: Universal.CustomNodataValue = None,
    image_parallel_workers: Universal.ImageParallelWorkers = None,
    window_parallel_workers: Universal.WindowParallelWorkers = None,
    window_size: Universal.WindowSize = None,
    custom_output_dtype: Universal.CustomOutputDtype = None,
    calculation_dtype: Universal.CalculationDtype = "float32",
):
    """
    Applies a thresholding operation to input raster images using a mathematical expression string.

    Args:
        input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_images (str | List[str], required): Defines output files from a template path, folder, or list of paths (with the same length as the input). Specify like: "/input/files/$.tif", "/input/folder" (assumes $_Threshold.tif), ["/input/one.tif", "/input/two.tif"].
        threshold_math (str): A logical expression string using bands (e.g., "b1 > 5", "b1 > 5 & b2 < 10"). Supports: Band references: b1, b2, ...; Operators: >, <, >=, <=, ==, !=, &, |, ~, and (); Percentile-based thresholds: use e.g. "5%b1" to use the 5th percentile of band 1.
        debug_logs (bool, optional): If True, prints debug messages.
        custom_nodata_value (float | int | None, optional): Override the dataset's nodata value.
        image_parallel_workers (ImageParallelWorkers, optional): Parallelism config for image-level processing.
        window_parallel_workers (WindowParallelWorkers, optional): Parallelism config for window-level processing.
        window_size (WindowSize, optional): Window tiling strategy for memory-efficient processing.
        custom_output_dtype (CustomOutputDtype, optional): Output data type override.
        calculation_dtype (CalculationDtype, optional): Internal computation dtype.
    """

    Universal.validate(
        input_images=input_images,
        output_images=output_images,
        debug_logs=debug_logs,
        custom_nodata_value=custom_nodata_value,
        image_parallel_workers=image_parallel_workers,
        window_parallel_workers=window_parallel_workers,
        window_size=window_size,
        custom_output_dtype=custom_output_dtype,
    )
    input_image_paths = _resolve_paths(
        "search", input_images, kwargs={"default_file_pattern": "*.tif"}
    )
    output_image_paths = _resolve_paths(
        "create",
        output_images,
        kwargs={
            "paths_or_bases": input_image_paths,
            "default_file_pattern": "$_Threshold.tif",
        },
    )
    image_names = _resolve_paths("name", input_image_paths)

    with rasterio.open(input_image_paths[0]) as ds:
        nodata_value = _resolve_nodata_value(ds, custom_nodata_value)
        output_dtype = _resolve_output_dtype(ds, custom_output_dtype)

    image_parallel, image_backend, image_max_workers = _resolve_parallel_config(
        image_parallel_workers
    )

    image_args = [
        (
            in_path,
            out_path,
            name,
            threshold_math,
            debug_logs,
            nodata_value,
            window_parallel_workers,
            window_size,
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
                executor.submit(_threshold_process_image, *arg) for arg in image_args
            ]
            for future in as_completed(futures):
                future.result()
    else:
        for arg in image_args:
            _threshold_process_image(*arg)


def _threshold_process_image(
    input_image_path: str,
    output_image_path: str,
    name: str,
    threshold_math: str,
    debug_logs: bool,
    nodata_value,
    window_parallel_workers,
    window_size,
    output_dtype,
    calculation_dtype,
):
    """
    Processes a single input raster image using a threshold expression and writes the result to disk.

    Args:
        input_image_path (str): Path to input raster image.
        output_image_path (str): Path to save the output thresholded image.
        name (str): Image name for worker context.
        threshold_math (str): Expression string to evaluate pixel-wise conditions.
        debug_logs (bool): Enable debug logging.
        nodata_value (float | int | None): Value considered as nodata.
        window_parallel_workers: Parallel config for window-level processing.
        window_size: Window tiling size for memory efficiency.
        output_dtype: Output raster data type.
        calculation_dtype: Data type used for internal calculations.
    """
    with rasterio.open(input_image_path) as src:
        profile = src.profile.copy()
        profile.update(
            dtype=output_dtype,
            count=1,
            nodata=nodata_value if nodata_value is not None else None,
        )

        window_parallel, window_backend, window_max_workers = _resolve_parallel_config(
            window_parallel_workers
        )

        percent_pattern = re.compile(r"(\d+(\.\d+)?)%b(\d+)")

        def replace_percent_with_threshold(match):
            percent, _, band_num = match.groups()
            value = _calculate_threshold_from_percent(
                input_image_path,
                f"{percent}%",
                int(band_num),
                debug_logs=debug_logs,
                nodata_value=nodata_value,
                window_parallel_workers=window_parallel_workers,
                window_size=window_size,
                calculation_dtype=calculation_dtype,
            )
            return str(value)

        evaluated_threshold_math = percent_pattern.sub(
            replace_percent_with_threshold, threshold_math
        )

        with rasterio.open(output_image_path, "w", **profile) as dst:
            windows = _resolve_windows(src, window_size)
            args = [
                (
                    name,
                    window,
                    evaluated_threshold_math,
                    debug_logs,
                    nodata_value,
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
                        executor.submit(_threshold_process_window, *arg) for arg in args
                    ]
                    for future in futures:
                        band, window, data = future.result()
                        dst.write(data.astype(output_dtype), band, window=window)
            else:
                WorkerContext.init({name: ("raster", input_image_path)})
                for arg in args:
                    band, window, data = _threshold_process_window(*arg)
                    dst.write(data.astype(output_dtype), band, window=window)
                WorkerContext.close()


def _threshold_process_window(
    name: str,
    window: rasterio.windows.Window,
    threshold_math: str,
    debug_logs: bool,
    nodata_value,
    calculation_dtype,
):
    """
    Applies the threshold logic to a single image window.

    Args:
        name (str): Image identifier for WorkerContext access.
        window (rasterio.windows.Window): Window to read and process.
        threshold_math (str): Logical expression for thresholding using b1, b2, etc.
        debug_logs (bool): Enable debug logs.
        nodata_value (float | int | None): Value considered as nodata.
        calculation_dtype: Dtype to cast bands for threshold computation.

    Returns:
        Tuple[int, rasterio.windows.Window, np.ndarray]: Band index, processed window, thresholded data mask (1 for true, 0 for false).
    """
    ds = WorkerContext.get(name)
    bands = {
        f"b{i+1}": ds.read(i + 1, window=window).astype(calculation_dtype)
        for i in range(ds.count)
    }

    if nodata_value is not None:
        nodata_mask = np.any([b == nodata_value for b in bands.values()], axis=0)
    else:
        nodata_mask = np.zeros_like(next(iter(bands.values())), dtype=bool)

    expr = threshold_math
    for k, v in bands.items():
        if isinstance(v, np.ndarray):
            expr = expr.replace(f"{k}", f"bands['{k}']")

    result = eval(expr, {"np": np, "bands": bands}).astype(calculation_dtype)
    result[nodata_mask] = nodata_value

    return 1, window, result


def _calculate_threshold_from_percent(
    input_image_path: str,
    threshold: str,
    band_index: int,
    *,
    debug_logs: bool = False,
    nodata_value=None,
    window_parallel_workers=None,
    window_size=None,
    calculation_dtype="float32",
    bins: int = 1000,
) -> float:
    """
    Calculates a threshold value based on a percentile of valid (non-nodata) pixel values in a raster.

    Args:
        input_image_path (str): Path to input raster image.
        threshold (str): Percent string (e.g., "5%") indicating the percentile to compute.
        band_index (int): Band index to evaluate.
        debug_logs (bool, optional): If True, prints debug info.
        nodata_value (float | int | None, optional): Value treated as nodata.
        window_parallel_workers: Optional parallel config.
        window_size: Tiling strategy.
        calculation_dtype (str): Internal dtype used for calculations.
        bins (int): Number of bins for histogram.

    Returns:
        float: Threshold value corresponding to the requested percentile.
    """

    percent = float(threshold.strip("%"))

    hist_total = np.zeros(bins, dtype=np.int64)
    min_val, max_val = None, None

    with rasterio.open(input_image_path) as src:
        windows = _resolve_windows(src, window_size)

        for window in windows:
            data = src.read(band_index, window=window).astype(calculation_dtype)
            if nodata_value is not None:
                data = data[data != nodata_value]
            if data.size == 0:
                continue

            win_min = data.min()
            win_max = data.max()
            min_val = win_min if min_val is None else min(min_val, win_min)
            max_val = win_max if max_val is None else max(max_val, win_max)

    if min_val is None or max_val is None or max_val <= min_val:
        raise ValueError("Unable to compute valid min/max range for histogram.")

    bin_range = (min_val, max_val)

    with rasterio.open(input_image_path) as src:
        windows = _resolve_windows(src, window_size)
        for window in windows:
            data = src.read(band_index, window=window).astype(calculation_dtype)
            if nodata_value is not None:
                data = data[data != nodata_value]
            if data.size == 0:
                continue

            hist, _ = np.histogram(data, bins=bins, range=bin_range)
            hist_total += hist

    cumsum = np.cumsum(hist_total)
    cutoff = (percent / 100.0) * cumsum[-1]
    bin_index = np.searchsorted(cumsum, cutoff)
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    value = bin_edges[min(bin_index, bins - 1)]

    if debug_logs:
        print(
            f"[threshold %] {threshold} → {value:.4f} using {bins} bins in range ({min_val:.4f}, {max_val:.4f})"
        )

    return value


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
        input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_vectors (str | List[str], required): Defines output files from a template path, folder, or list of paths (with the same length as the input). Specify like: "/input/files/$.gpkg", "/input/folder" (assumes $_Vectorized.gpkg), ["/input/one.gpkg", "/input/two.gpkg"].
        custom_nodata_value (Universal.CustomNodataValue, optional): Custom NoData value to override the default from the raster metadata.
        custom_output_dtype (Universal.CustomOutputDtype, optional): Desired output data type. If not set, defaults to raster’s dtype.
        image_parallel_workers (Universal.ImageParallelWorkers, optional): Controls parallelism across input images. Can be an integer, executor string, or boolean.
        window_parallel_workers (Universal.WindowParallelWorkers, optional): Controls parallelism within a single image by processing windows in parallel.
        window_size (Universal.WindowSizeWithBlock, optional): Size of each processing block (width, height), or a strategy string such as "block" or "whole".
        debug_logs (Universal.DebugLogs, optional): Whether to print debug logs to the console.
        extraction_expression (str): Logical expression to identify pixels of interest using band references (e.g., "b1 > 10 & b2 < 50").
        filter_by_polygon_size (str, optional): Area filter for resulting polygons. Can be a number (e.g., ">100") or percentile (e.g., ">95%").
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

    input_image_paths = _resolve_paths(
        "search", input_images, kwargs={"default_file_pattern": "*.tif"}
    )
    output_image_paths = _resolve_paths(
        "create",
        output_vectors,
        kwargs={
            "paths_or_bases": input_image_paths,
            "default_file_pattern": "$_Vectorized.gpkg",
        },
    )

    image_parallel, image_backend, image_max_workers = _resolve_parallel_config(
        image_parallel_workers
    )
    window_parallel, window_backend, window_max_workers = _resolve_parallel_config(
        window_parallel_workers
    )

    image_args = [
        (
            in_path,
            out_path,
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
        )
        for in_path, out_path in zip(input_image_paths, output_image_paths)
    ]

    if image_parallel:
        with _get_executor(image_backend, image_max_workers) as executor:
            futures = [
                executor.submit(_process_image_to_polygons, *args)
                for args in image_args
            ]
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

        band_indices = sorted(
            set(int(b[1:]) for b in re.findall(r"b\d+", extraction_expression))
        )
        band_indices = sorted(set(band_indices))

        windows = _resolve_windows(src, window_size)
        window_args = [
            (w, band_indices, extraction_expression, value_mapping, nodata_value)
            for w in windows
        ]

        polygons = []
        if window_parallel:
            with _get_executor(
                window_backend,
                window_max_workers,
                initializer=WorkerContext.init,
                initargs=({"input": ("raster", input_image_path)},),
            ) as executor:
                futures = [
                    executor.submit(_process_window, *args) for args in window_args
                ]
                for f in as_completed(futures):
                    polygons.extend(f.result())
        else:
            WorkerContext.init({"input": ("raster", input_image_path)})
            for args in window_args:
                polygons.extend(_process_window(*args))
            WorkerContext.close()

    if not polygons:
        if debug_logs:
            print("No features found.")
        return

    gdf = gpd.GeoDataFrame(polygons, crs=crs)
    merged = gdf.dissolve(by="value", as_index=False)

    if filter_by_polygon_size:
        match = re.match(
            r"([<>]=?|==|!=)\s*(\d+(?:\.\d+)?%?)", filter_by_polygon_size.strip()
        )
        if not match:
            raise ValueError(
                f"Invalid filter_by_polygon_size format: {filter_by_polygon_size}"
            )

        op, val = match.groups()

        if val.endswith("%"):
            pct = float(val.strip("%"))
            area_thresh = np.percentile(merged.geometry.area, pct)
        else:
            area_thresh = float(val)

        op_map = {
            "<": lambda x: x < area_thresh,
            "<=": lambda x: x <= area_thresh,
            ">": lambda x: x > area_thresh,
            ">=": lambda x: x >= area_thresh,
            "==": lambda x: x == area_thresh,
            "!=": lambda x: x != area_thresh,
        }
        merged = merged[op_map[op](merged.geometry.area)]

    if polygon_buffer:
        merged["geometry"] = merged.geometry.buffer(polygon_buffer)

    if os.path.exists(output_vector_path):
        os.remove(output_vector_path)

    merged.to_file(output_vector_path, driver="GPKG", layer="mask")


def _process_window(window, band_indices, expression, value_mapping, nodata_value):
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
