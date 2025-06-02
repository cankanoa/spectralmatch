import os

import fiona
import numpy as np
import tempfile
import rasterio
import shutil
import geopandas as gpd
import glob
import pandas as pd
import re
import warnings

from typing import List, Optional, Literal, Tuple
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform, reproject
from rasterio.coords import BoundingBox
from multiprocessing import Pool, cpu_count, get_context
from concurrent.futures import as_completed
from rasterio.transform import Affine

from .types_and_validation import Universal
from .utils_multiprocessing import _create_windows, _resolve_windows, _get_executor, WorkerContext, _resolve_parallel_config


def merge_vectors(
        input_vector_paths: List[str],
        merged_vector_path: str,
        method: Literal["intersection", "union", "keep_all"],
        debug_logs: bool = False,
        create_name_attribute: Optional[Tuple[str, str]] = None,
) -> None:
    """
    Merge multiple vector files using the specified geometric method.

    Args:
        input_vector_paths (List[str]): Paths to input vector files.
        merged_vector_path (str): Path to save merged output.
        method (Literal["intersection", "union", "keep_all"]): Merge strategy.
        debug_logs (bool): If True, print debug information.
        create_name_attribute (Optional[Tuple[str, str]]): Tuple of (field_name, separator).
            If set, adds a field with all input filenames (no extension), joined by separator.

    Returns:
        None
    """
    print(f"Start vector merge")

    os.makedirs(os.path.dirname(merged_vector_path), exist_ok=True)

    geoms = []
    input_names = []

    for path in input_vector_paths:
        gdf = gpd.read_file(path)
        if create_name_attribute:
            name = os.path.splitext(os.path.basename(path))[0]
            input_names.append(name)
        geoms.append(gdf)

    # Prepare the full combined name value once
    combined_name_value = None
    if create_name_attribute:
        field_name, sep = create_name_attribute
        combined_name_value = sep.join(input_names)

    if method == "keep_all":
        merged = gpd.GeoDataFrame(pd.concat(geoms, ignore_index=True), crs=geoms[0].crs)
        if create_name_attribute:
            merged[field_name] = combined_name_value

    elif method == "union":
        merged = gpd.GeoDataFrame(pd.concat(geoms, ignore_index=True), crs=geoms[0].crs)
        if create_name_attribute:
            merged[field_name] = combined_name_value

    elif method == "intersection":
        merged = geoms[0]
        for gdf in geoms[1:]:
            shared_cols = set(merged.columns).intersection(gdf.columns) - {"geometry"}
            gdf = gdf.drop(columns=shared_cols)
            merged = gpd.overlay(merged, gdf, how="intersection", keep_geom_type=True)
        if create_name_attribute:
            merged[field_name] = combined_name_value

    else:
        raise ValueError(f"Unsupported merge method: {method}")

    merged.to_file(merged_vector_path)


def _merge_tile(
    args: Tuple[
        Window,
        rasterio.Affine,
        Tuple[int, int],
        List[str],
        int,
        str,
        float | int,
        str,
    ],
) -> Tuple[Window, np.ndarray]:
    window, transform, shape, image_paths, band_count, dtype, nodata, dst_crs = args
    tile_data = np.full((band_count, shape[1], shape[0]), nodata, dtype=dtype)

    for path in image_paths:
        with rasterio.open(path) as src:
            for b in range(band_count):
                temp = np.full((shape[1], shape[0]), nodata, dtype=dtype)
                reproject(
                    source=rasterio.band(src, b + 1),
                    destination=temp,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                    dst_nodata=nodata,
                )
                # Only fill in values where current tile is still nodata
                mask = (tile_data[b] == nodata) & (temp != nodata)
                tile_data[b][mask] = temp[mask]

    return window, tile_data


def align_rasters(
    input_images: Universal.SearchFolderOrListFiles,
    output_images: Universal.CreateInFolderOrListFiles,
    *,
    resampling_method: Literal["nearest", "bilinear", "cubic"] = "bilinear",
    tap: bool = False,
    resolution: Literal["highest", "average", "lowest"] = "highest",
    window_size: Universal.WindowSize = None,
    debug_logs: Universal.DebugLogs = False,
    image_parallel_workers: Universal.ImageParallelWorkers = None,
    window_parallel_workers: Universal.WindowParallelWorkers = None,
    ) -> None:

    print("Start align rasters")

    input_image_paths = _resolve_paths("search", input_images)
    output_image_paths = _resolve_paths("create", output_images, (input_image_paths,))
    image_names = os.path.splitext(os.path.basename(input_image_paths[0]))[0]

    if debug_logs: print(f"{len(input_image_paths)} rasters to align")

    # Determine target resolution
    resolutions = []
    crs_list = []
    for path in input_image_paths:
        with rasterio.open(path) as src:
            resolutions.append(src.res)
            crs_list.append(src.crs)
    if len(set(crs_list)) > 1:
        raise ValueError("Input rasters must have the same CRS.")

    res_arr = np.array(resolutions)
    target_res = {
        "highest": res_arr.min(axis=0),
        "lowest": res_arr.max(axis=0),
        "average": res_arr.mean(axis=0),
    }[resolution]

    if debug_logs: print(f"Target resolution: {target_res}")

    parallel_args = [
        (
            image_name,
            window_parallel_workers,
            in_path,
            out_path,
            target_res,
            resampling_method,
            tap,
            window_size,
            debug_logs,
        )
        for in_path, out_path, image_name in zip(input_image_paths, output_image_paths, image_names)
    ]

    if image_parallel_workers:
        with _get_executor(*image_parallel_workers) as executor:
            futures = [executor.submit(_align_process_image, *args) for args in parallel_args]
            for future in as_completed(futures):
                future.result()
    else:
        for args in parallel_args:
            _align_process_image(*args)


def _align_process_image(
    image_name: str,
    window_parallel: Universal.WindowParallelWorkers,
    in_path: str,
    out_path: str,
    target_res: Tuple[float, float],
    resampling_method: str,
    tap: bool,
    window_size: Universal.WindowSize,
    debug_logs: bool,
):
    if debug_logs: print(f"Aligning: {in_path}")

    with rasterio.open(in_path) as src:
        profile = src.profile.copy()

        if tap:
            res_x, res_y = target_res
            minx = np.floor(src.bounds.left / res_x) * res_x
            miny = np.floor(src.bounds.bottom / res_y) * res_y
            maxx = np.ceil(src.bounds.right / res_x) * res_x
            maxy = np.ceil(src.bounds.top / res_y) * res_y
            dst_width = int((maxx - minx) / res_x)
            dst_height = int((maxy - miny) / res_y)
            dst_transform = rasterio.transform.from_origin(minx, maxy, res_x, res_y)
        else:
            dst_width, dst_height = src.width, src.height
            dst_transform = src.transform

        src_transform = src.transform

        profile.update({
            "height": dst_height,
            "width": dst_width,
            "transform": dst_transform
        })

        with rasterio.open(out_path, "w", **profile) as dst:
            for band_idx in range(src.count):

                windows_dst = _resolve_windows(dst, window_size)

                window_args = []
                for dst_win in windows_dst:
                    dst_bounds = rasterio.windows.bounds(dst_win, dst.transform)

                    # Convert bounds to source window using inverse transform
                    src_win = rasterio.windows.from_bounds(*dst_bounds, transform=src.transform)
                    # src_win = src_win.round_offsets().round_lengths()  # Makes it integer-aligned

                    window_args.append((
                        src_win,
                        dst_win,
                        band_idx,
                        dst_transform,
                        resampling_method,
                        src.nodata,
                        debug_logs,
                        image_name,
                    ))

                parallel = window_parallel is not None
                backend, max_workers = (window_parallel or (None, None))[0:2]

                if parallel and backend == "process":
                    with _get_executor(
                            backend,
                            max_workers,
                            initializer=WorkerContext.init,
                            initargs=({image_name: ("raster", in_path)},)
                    ) as executor:
                        futures = [executor.submit(_align_process_window, *args) for args in window_args]
                        for future in as_completed(futures):
                            band, window, buf = future.result()
                            dst.write(buf, band + 1, window=window)
                    WorkerContext.close()
                else:
                    WorkerContext.init({image_name: ("raster", in_path)})
                    for args in window_args:
                        band, window, buf = _align_process_window(*args)
                        dst.write(buf, band + 1, window=window)
                    WorkerContext.close()


def _align_process_window(
    src_window: Window,
    dst_window: Window,
    band_idx: int,
    dst_transform,
    resampling_method: str,
    nodata: int | float,
    debug_logs: bool,
    image_name: str,
) -> tuple[int, Window, np.ndarray]:
    """
    Aligns a single raster window for one band using reproject with a shared dataset.

    Args:
        src_window (Window): Source window to read.
        dst_window (Window): Output window (used to compute offset transform and for saving).
        band_idx (int): Band index to read.
        dst_transform: The full transform of the output raster.
        resampling_method: Reprojection resampling method.
        nodata: NoData value.
        debug_logs: Print debug info if True.
        image_name: Key to fetch the raster from WorkerContext.

    Returns:
        Tuple[int, Window, np.ndarray]: Band index, destination window, and aligned data buffer.
    """
    src = WorkerContext.get(image_name)
    dst_shape = (int(dst_window.height), int(dst_window.width))
    dst_buffer = np.empty(dst_shape, dtype=src.dtypes[band_idx])

    # Compute the transform specific to the current dst_window tile
    dst_transform_window = dst_transform * Affine.translation(dst_window.col_off, dst_window.row_off)

    reproject(
        source=rasterio.band(src, band_idx + 1),
        destination=dst_buffer,
        src_transform=src.window_transform(src_window),
        src_crs=src.crs,
        dst_transform=dst_transform_window,
        dst_crs=src.crs,
        src_nodata=nodata,
        dst_nodata=nodata,
        resampling=Resampling[resampling_method],
        src_window=src_window,
        dst_window=Window(0, 0, dst_shape[1], dst_shape[0]),
    )

    return band_idx, dst_window, dst_buffer


def merge_rasters(
    input_images: Tuple[str, str] | List[str],
    output_image_path: str,
    window_size: Optional[int | Tuple[int, int]] = None,
    parallel_workers: Literal["cpu"] | int | None = None,
    debug_logs: bool = False,
    output_dtype: str | None = None,
    custom_nodata_value: float | int | None = None,
) -> None:
    """
    Merge multiple rasters efficiently using tile-based multiprocessing.

    Args:
        input_images (Tuple[str, str] | List[str]): List or tuple of input paths. Tuple triggers search_paths(*input_images).
        output_image_path (str): Path to save the merged raster.
        window_size (int | Tuple[int, int] | None): Tile size (width, height). If None, full image is used.
        parallel_workers (Literal["cpu"] | int | None): Number of workers. Use "cpu" to use all cores.
        debug_logs (bool): Enable debug logging.
        output_dtype (str | None): Output data type. Defaults to first image dtype.
        custom_nodata_value (float | int | None): Optional nodata override.

    Returns:
        None
    """
    print('Start merge')

    if isinstance(input_images, tuple):
        input_images = search_paths(*input_images)

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    if isinstance(window_size, int):
        window_size = (window_size, window_size)

    with rasterio.open(input_images[0]) as ref:
        dst_crs = ref.crs
        band_count = ref.count
        ref_dtype = ref.dtypes[0]
        ref_nodata = ref.nodata
        output_dtype = output_dtype or ref_dtype
        nodata = custom_nodata_value if custom_nodata_value is not None else ref_nodata

    if debug_logs:
        print(f"Calculating output bounds and resolution...")

    # Initialize union bounds
    dst_bounds = None
    dst_resolution = None

    for path in input_images:
        with rasterio.open(path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            bounds = rasterio.transform.array_bounds(height, width, transform)
            bbox = BoundingBox(*bounds)
            if dst_bounds is None:
                dst_bounds = bbox
            else:
                dst_bounds = BoundingBox(
                    left=min(dst_bounds.left, bbox.left),
                    bottom=min(dst_bounds.bottom, bbox.bottom),
                    right=max(dst_bounds.right, bbox.right),
                    top=max(dst_bounds.top, bbox.top),
                )
            if dst_resolution is None:
                dst_resolution = (abs(transform.a), abs(transform.e))

    dst_transform = rasterio.transform.from_origin(dst_bounds.left, dst_bounds.top, *dst_resolution)
    dst_width = int(np.ceil((dst_bounds.right - dst_bounds.left) / dst_resolution[0]))
    dst_height = int(np.ceil((dst_bounds.top - dst_bounds.bottom) / dst_resolution[1]))

    if debug_logs:
        print(f"Output size: {dst_width}x{dst_height}")
        print(f"Window size: {window_size}")

    meta = {
        "driver": "GTiff",
        "width": dst_width,
        "height": dst_height,
        "count": band_count,
        "dtype": output_dtype,
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": nodata,
        "tiled": True,
        "blockxsize": window_size[0],
        "blockysize": window_size[1],
        "compress": "deflate",
    }

    def window_grid():
        for row_off in range(0, dst_height, window_size[1]):
            for col_off in range(0, dst_width, window_size[0]):
                win = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=min(window_size[0], dst_width - col_off),
                    height=min(window_size[1], dst_height - row_off),
                )
                yield (
                    win,
                    rasterio.windows.transform(win, dst_transform),
                    (win.width, win.height),
                    input_images,
                    band_count,
                    output_dtype,
                    nodata,
                    dst_crs,
                )

    windows = list(window_grid())

    if debug_logs:
        print(f"Processing {len(windows)} tiles")

    num_workers = cpu_count() if parallel_workers == "cpu" else parallel_workers or 1

    with rasterio.open(output_image_path, "w", **meta) as dst:
        with get_context("spawn").Pool(num_workers) as pool:
            for win, data in pool.imap(_merge_tile, windows):
                dst.write(data, window=win)

    if debug_logs: print(f"Saved merged raster: {output_image_path}")


def mask_rasters(
    input_images: Universal.SearchFolderOrListFiles,
    output_images: Universal.CreateInFolderOrListFiles,
    vector_mask: Universal.VectorMask = None,
    window_size: Universal.WindowSize = None,
    debug_logs: Universal.DebugLogs = False,
    image_parallel_workers: Universal.ImageParallelWorkers = None,
    window_parallel_workers: Universal.WindowParallelWorkers = None,
    include_touched_pixels: bool = False,
    ) -> None:
    """
    Masks rasters using optional vector geometries with support for window-based and image-based multiprocessing.
    """
    # Validate parameters
    Universal.validate(
        input_images=input_images,
        output_images=output_images,
        debug_logs=debug_logs,
        vector_mask=vector_mask,
        window_size=window_size,
        image_parallel_workers=image_parallel_workers,
        window_parallel_workers=window_parallel_workers,
    )

    input_image_paths = _resolve_paths("search", input_images)
    output_image_paths = _resolve_paths("create", output_images, (input_image_paths,))

    if debug_logs: print(f"Input images: {input_image_paths}")
    if debug_logs: print(f"Output images: {output_image_paths}")

    input_image_names = [os.path.splitext(os.path.basename(p))[0] for p in input_image_paths]
    input_image_path_pairs = dict(zip(input_image_names, input_image_paths))
    output_image_path_pairs = dict(zip(input_image_names, output_image_paths))

    image_parallel, image_backend, image_max_workers = _resolve_parallel_config(image_parallel_workers)
    window_parallel, window_backend, window_max_workers = _resolve_parallel_config(window_parallel_workers)

    parallel_args = [
        (
            window_parallel,
            window_max_workers,
            window_backend,
            input_image_path_pairs[name],
            output_image_path_pairs[name],
            name,
            vector_mask,
            window_size,
            debug_logs,
            include_touched_pixels
        )
        for name in input_image_names
    ]

    if image_parallel:
        with _get_executor(image_backend, image_max_workers) as executor:
            futures = [executor.submit(_mask_raster_process_image, *args) for args in parallel_args]
            for future in as_completed(futures):
                future.result()
    else:
        for args in parallel_args:
            _mask_raster_process_image(*args)


def _mask_raster_process_image(
    window_parallel: bool,
    max_workers: int,
    backend: str,
    input_image_path: str,
    output_image_path: str,
    image_name: str,
    vector_mask: Universal.VectorMask,
    window_size: Universal.WindowSize,
    debug_logs: bool,
    include_touched_pixels: bool,
):
    with rasterio.open(input_image_path) as src:
        profile = src.profile.copy()
        nodata_val = profile.get("nodata", 0)
        num_bands = src.count

        geoms = None
        invert = False

        if vector_mask:
            mode, path, *field = vector_mask
            invert = mode == "exclude"
            field_name = field[0] if field else None
            with fiona.open(path, "r") as vector:
                if field_name:
                    geoms = [
                        feat["geometry"]
                        for feat in vector
                        if field_name in feat["properties"] and image_name in str(feat["properties"][field_name])
                    ]
                else:
                    geoms = [feat["geometry"] for feat in vector]

        with rasterio.open(output_image_path, "w", **profile) as dst:

            for band_idx in range(num_bands):
                windows = _resolve_windows(src, window_size)

                args = [
                    (
                        win,
                        band_idx,
                        image_name,
                        nodata_val,
                        geoms,
                        invert,
                        include_touched_pixels
                    )
                    for win in windows
                ]

                if window_parallel:
                    with _get_executor(
                        backend,
                        max_workers,
                        initializer=WorkerContext.init,
                        initargs=({image_name: ("raster", input_image_path)},)
                    ) as executor:
                        futures = [executor.submit(_mask_raster_process_window, *arg) for arg in args]
                        for future in as_completed(futures):
                            window, data = future.result()
                            dst.write(data, band_idx + 1, window=window)
                    WorkerContext.close()
                else:
                    WorkerContext.init({image_name: ("raster", input_image_path)})
                    for arg in args:
                        window, data = _mask_raster_process_window(*arg)
                        dst.write(data, band_idx + 1, window=window)
                    WorkerContext.close()



def _mask_raster_process_window(
    win: Window,
    band_idx: int,
    image_name: str,
    nodata: int | float,
    geoms: list | None,
    invert: bool,
    include_touched_pixels: bool,
):
    src = WorkerContext.get(image_name)
    mask_key = f"{int(win.col_off)}-{int(win.row_off)}-{int(win.width)}-{int(win.height)}"

    mask_cache = WorkerContext.cache.setdefault("_mask_cache", {})
    mask_hits = WorkerContext.cache.setdefault("_mask_hits", {})

    if geoms:
        if mask_key not in mask_cache:
            transform = src.window_transform(win)
            mask = geometry_mask(
                geoms,
                transform=transform,
                invert=not invert,
                out_shape=(int(win.height), int(win.width)),
                all_touched=include_touched_pixels
            )
            mask_cache[mask_key] = mask
            mask_hits[mask_key] = 0

        mask = mask_cache[mask_key]
        data = src.read(band_idx + 1, window=win)
        data = np.where(mask, data, nodata)

        # Track usage and clean up
        mask_hits[mask_key] += 1
        if mask_hits[mask_key] >= src.count:
            del mask_cache[mask_key]
            del mask_hits[mask_key]
    else:
        data = src.read(band_idx + 1, window=win)

    return win, data


# def _mask_raster_process_window(
#     win: Window,
#     image_name: str,
#     nodata: int | float,
#     geoms: list | None,
#     invert: bool,
#     include_touched_pixels: bool,
# ):
#     src = WorkerContext.get(image_name)
#     data = src.read(window=win)
#
#     if geoms:
#         transform = src.window_transform(win)
#         mask = geometry_mask(
#             geoms,
#             transform=transform,
#             invert=not invert,
#             out_shape=(data.shape[1], data.shape[2]),
#             all_touched=include_touched_pixels
#         )
#         data = np.where(mask, data, nodata)
#
#     return win, data


def _resolve_paths(
    mode: Literal["search", "create", "match"],
    input: Universal.SearchFolderOrListFiles | Universal.CreateInFolderOrListFiles,
    args: Tuple | None = None,
) -> List[str]:
    """
    Resolves a list of input based on the mode and input format.

    Args:
        mode (Literal["search", "create", "match"]): Type of operation to perform.
        input (Tuple[str, str] | List[str]): Either a list of file input or a tuple specifying folder/template info.
        args (Tuple): Additional arguments passed to the called function.

    Returns:
        List[str]: List of resolved input.
    """
    if isinstance(input, list):
        resolved = input
    elif mode == "search":
        resolved = search_paths(input[0], input[1], *(args or ()))
    elif mode == "create":
        resolved = create_paths(input[0], input[1], *(args or ()))
    elif mode == "match":
        resolved = match_paths(*(args or ()))
    else: raise ValueError(f"Invalid mode: {mode}")

    if len(resolved) == 0:
        warnings.warn(f"No results found for paths.", RuntimeWarning)

    return resolved

def search_paths(
    folder_path: str,
    pattern: str,
    recursive: bool = False,
    match_to_paths: Tuple[List[str], str] | None = None,
    debug_logs: bool = False,
    ) -> List[str]:
    """
    Search for files in a folder using a glob pattern.

    Args:
        folder_path (str): The root folder to search in.
        pattern (str): A glob pattern (e.g., "*.tif", "**/*.jpg").
        recursive (bool, optional): Whether to search for files recursively.
        match_to_paths (Tuple[List[str], str], optional): If provided, match `reference_paths` to `input_match_paths` using a regex applied to the basenames of `input_match_paths`. The extracted key must be a substring of the reference filename.
         - reference_paths (List[str]): List of reference paths to align to.
         - match_regex (str): Regex applied to basenames of input_match_paths to extract a key to match via *inclusion* in reference_paths (e.g. r"(.*)_LocalMatch\.gpkg$").
        debug_logs (bool, optional): Whether to print the matched file paths.

    Returns:
        List[str]: Sorted list of matched file paths.
    """
    input_paths =  sorted(glob.glob(os.path.join(folder_path, pattern), recursive=recursive))

    if match_to_paths:
        input_paths = match_paths(input_paths, *match_to_paths)

    return input_paths

def create_paths(
    output_folder: str,
    template: str,
    paths_or_bases: List[str],
    debug_logs: bool = False,
    replace_symbol: str = "$",
    create_folders: bool = True,
    ) -> List[str]:
    """
    Create output paths using a filename template and a list of reference paths or names.

    Args:
        output_folder (str): Directory to store output files.
        template (str): Filename template using replace_symbol as placeholder (e.g., "$_processed.tif").
        paths_or_bases (List[str]): List of full paths or bare names to derive replace_symbol from. Inclusion of '/' or '\' indicates a path.
        debug_logs (bool): Whether to print the created paths.
        replace_symbol (str): Symbol to replace in the template.
        create_folders (bool): Whether to create output folders if they don't exist.'

    Returns:
        List[str]: List of constructed file paths.
    """
    output_paths = []
    for ref in paths_or_bases:
        base = os.path.splitext(os.path.basename(ref))[0] if ('/' in ref or '\\' in ref) else os.path.splitext(ref)[0]
        filename = template.replace(replace_symbol, base)
        path = os.path.join(output_folder, filename)
        output_paths.append(path)

    if create_folders:
        for path in output_paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)
    return output_paths


def match_paths(
    input_match_paths: List[str],
    reference_paths: List[str],
    match_regex: str,
    debug_logs: bool = False,
    ) -> List[Optional[str]]:
    """
    Match `reference_paths` to `input_match_paths` using a regex applied to the basenames of `input_match_paths`. The extracted key must be a substring of the reference filename.

    Args:
        input_match_paths (List[str]): List of candidate paths to extract keys from.
        reference_paths (List[str]): List of reference paths to align to.
        match_regex (str): Regex applied to basenames of input_match_paths to extract a key to match via *inclusion* in reference_paths (e.g. r\"(.*)_LocalMatch\.gpkg$").
        debug_logs (bool): If True, print matched and unmatched file basenames.

    Returns:
        List[Optional[str]]: A list the same length as `reference_paths` where each
        element is the matched path from `input_match_paths` or None.

    Raises:
        ValueError: If output list length does not match reference_paths length.
    """
    pattern = re.compile(match_regex)
    match_keys = {}
    used_matches = set()

    # Extract keys from input_match_paths
    for mpath in input_match_paths:
        basename = os.path.basename(mpath)
        match = pattern.search(basename)
        if not match:
            continue
        key = match.group(1) if match.groups() else match.group(0)
        match_keys[key] = mpath

    # Match each reference path
    matched_list: List[Optional[str]] = []
    for rpath in reference_paths:
        rbase = os.path.basename(rpath)
        matched = None
        for key, mpath in match_keys.items():
            if key in rbase:
                matched = mpath
                used_matches.add(mpath)
                break
        matched_list.append(matched)

    # Validate output length
    if len(matched_list) != len(reference_paths):
        raise ValueError("Matched list length does not match reference_paths length.")

    return matched_list