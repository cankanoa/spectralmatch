import os
import rasterio
import numpy as np
import re

from concurrent.futures import as_completed

from ..utils_multiprocessing import _get_executor, WorkerContext, _resolve_windows, _resolve_parallel_config
from ..handlers import _resolve_paths, _resolve_nodata_value, _resolve_output_dtype
from ..types_and_validation import Universal


def threshold_raster(
    input_images: Universal.SearchFolderOrListFiles,
    output_images: Universal.CreateInFolderOrListFiles,
    operator: str,
    threshold: float | str,
    *,
    band_index: int = 1,
    debug_logs: Universal.DebugLogs = False,
    custom_nodata_value: Universal.CustomNodataValue = None,
    image_parallel_workers: Universal.ImageParallelWorkers = None,
    window_parallel_workers: Universal.WindowParallelWorkers = None,
    window_size: Universal.WindowSize = None,
    custom_output_dtype: Universal.CustomOutputDtype = None,
    calculation_dtype: Universal.CalculationDtype = "float32",
):
    input_image_paths = _resolve_paths("search", input_images)
    output_image_paths = _resolve_paths("create", output_images, (input_image_paths,))
    image_names = _resolve_paths("name", input_image_paths)

    with rasterio.open(input_image_paths[0]) as ds:
        nodata_value = _resolve_nodata_value(ds, custom_nodata_value)
        output_dtype = _resolve_output_dtype(ds, custom_output_dtype)

    image_parallel, image_backend, image_max_workers = _resolve_parallel_config(image_parallel_workers)

    image_args = [
        (in_path, out_path, name, operator, threshold, band_index, debug_logs, nodata_value,
         window_parallel_workers, window_size, output_dtype, calculation_dtype)
        for in_path, out_path, name in zip(input_image_paths, output_image_paths, image_names)
    ]

    if image_parallel:
        with _get_executor(image_backend, image_max_workers) as executor:
            futures = [executor.submit(_threshold_process_image, *arg) for arg in image_args]
            for future in as_completed(futures):
                future.result()
    else:
        for arg in image_args:
            _threshold_process_image(*arg)


def _threshold_process_image(
    input_image_path: str,
    output_image_path: str,
    name: str,
    operator: str,
    threshold: float,
    band_index: int | str,
    debug_logs: bool,
    nodata_value,
    window_parallel_workers,
    window_size,
    output_dtype,
    calculation_dtype,
):
    with rasterio.open(input_image_path) as src:
        if isinstance(threshold, str) and threshold.endswith('%'):
            threshold = _calculate_threshold_from_percent(
                input_image_path,
                threshold,
                band_index,
                debug_logs=debug_logs,
                nodata_value=nodata_value,
                window_parallel_workers=window_parallel_workers,
                window_size=window_size,
                calculation_dtype=calculation_dtype,
            )

        profile = src.profile.copy()
        profile.update(dtype=output_dtype, count=1)

        window_parallel, window_backend, window_max_workers = _resolve_parallel_config(window_parallel_workers)

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        with rasterio.open(output_image_path, "w", **profile) as dst:
            windows = _resolve_windows(src, window_size)
            args = [
                (name, window, operator, threshold, band_index, debug_logs, nodata_value, calculation_dtype)
                for window in windows
            ]

            if window_parallel:
                with _get_executor(window_backend, window_max_workers,
                                   initializer=WorkerContext.init,
                                   initargs=({name: ("raster", input_image_path)},)) as executor:
                    futures = [executor.submit(_threshold_process_window, *arg) for arg in args]
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
    operator: str,
    threshold: float,
    band_index: int,
    debug_logs: bool,
    nodata_value,
    calculation_dtype
):
    ds = WorkerContext.get(name)
    band = ds.read(band_index, window=window).astype(calculation_dtype)

    if nodata_value is not None:
        nodata_mask = band == nodata_value
    else:
        nodata_mask = np.zeros_like(band, dtype=bool)

    try:
        expr = f"band {operator} {threshold}"
        result = eval(expr, {"band": band, "np": np}).astype(calculation_dtype)
        result[nodata_mask] = np.nan
    except Exception as e:
        raise ValueError(f"Failed threshold operation '{expr}': {e}")

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

    assert threshold.endswith('%'), f"Invalid threshold format: {threshold}"
    percent = float(threshold.strip('%'))

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
        print(f"[threshold %] {threshold} → {value:.4f} using {bins} bins in range ({min_val:.4f}, {max_val:.4f})")

    return value


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
    input_image_paths = _resolve_paths("search", input_images)
    output_image_paths = _resolve_paths("create", output_images, (input_image_paths,))
    image_names = _resolve_paths("name", input_image_paths)

    with rasterio.open(input_image_paths[0]) as ds:
        nodata_value = _resolve_nodata_value(ds, custom_nodata_value)
        output_dtype = _resolve_output_dtype(ds, custom_output_dtype)

    image_parallel, image_backend, image_max_workers = _resolve_parallel_config(image_parallel_workers)

    # Extract referenced bands from custom_math (e.g., b1, b2, ...)
    band_indices = sorted({int(match[1:]) for match in re.findall(r"\bb\d+\b", custom_math)})

    image_args = [
        (in_path, out_path, name, custom_math, debug_logs, nodata_value, window_parallel_workers, window_size, band_indices, output_dtype, calculation_dtype)
        for in_path, out_path, name in zip(input_image_paths, output_image_paths, image_names)
    ]

    if image_parallel:
        with _get_executor(image_backend, image_max_workers) as executor:
            futures = [executor.submit(band_math_process_image, *arg) for arg in image_args]
            for future in as_completed(futures):
                future.result()
    else:
        for arg in image_args:
            band_math_process_image(*arg)


def band_math_process_image(
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
    with rasterio.open(input_image_path) as src:
        profile = src.profile.copy()
        profile.update(dtype=output_dtype, count=1)

        window_parallel, window_backend, window_max_workers = _resolve_parallel_config(window_parallel_workers)

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        with rasterio.open(output_image_path, "w", **profile) as dst:
            windows = _resolve_windows(src, window_size)
            args = [
                (name, window, custom_math, debug_logs, nodata_value, band_indices, calculation_dtype)
                for window in windows
            ]

            if window_parallel:
                with _get_executor(window_backend, window_max_workers, initializer=WorkerContext.init, initargs=({name: ("raster", input_image_path)},)) as executor:
                    futures = [executor.submit(band_math_process_window, *arg) for arg in args]
                    for future in futures:
                        band, window, data = future.result()
                        dst.write(data.astype(output_dtype), band, window=window)
            else:
                WorkerContext.init({name: ("raster", input_image_path)})
                for arg in args:
                    band, window, data = band_math_process_window(*arg)
                    dst.write(data.astype(output_dtype), band, window=window)
                WorkerContext.close()


def band_math_process_window(
    name: str,
    window: rasterio.windows.Window,
    custom_math: str,
    debug_logs: bool,
    nodata_value,
    band_indices,
    calculation_dtype
):
    ds = WorkerContext.get(name)

    bands = [ds.read(i, window=window).astype(calculation_dtype) for i in band_indices]
    band_vars = {f"b{i}": b for i, b in zip(band_indices, bands)}

    try:
        result = eval(custom_math, {"np": np}, band_vars).astype(calculation_dtype)
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression '{custom_math}': {e}")

    if nodata_value is not None:
        nodata_mask = np.any([b == nodata_value for b in bands], axis=0)
        result[nodata_mask] = np.nan

    return 1, window, result
