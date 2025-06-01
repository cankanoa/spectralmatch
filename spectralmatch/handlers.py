import os
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

from .types_and_validation import Universal
from .utils_multiprocessing import _create_windows, _resolve_windows, _get_executor, WorkerContext


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
    resampling_method: Literal["nearest", "bilinear", "cubic"] = "nearest",
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
            window_parallel_workers,
            in_path,
            out_path,
            target_res,
            resampling_method,
            tap,
            window_size,
            debug_logs,
        )
        for in_path, out_path in zip(input_image_paths, output_image_paths)
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

        profile.update({
            "height": dst_height,
            "width": dst_width,
            "transform": dst_transform
        })

        windows_src = _resolve_windows(src, window_size)
        windows_dst = _resolve_windows(
            dataset=type("FakeDS", (), {
                "width": dst_width,
                "height": dst_height,
                "transform": dst_transform,
                "block_windows": lambda _: [],
                "bounds": src.bounds
            })(),
            window_size=window_size,
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            window_args = [
                (
                    src_win,
                    dst_win,
                    dst_transform,
                    resampling_method,
                    "src_key",
                    (dst_win.height, dst_win.width),
                    debug_logs,
                )
                for src_win, dst_win in zip(windows_src, windows_dst)
            ]

            parallel = window_parallel is not None
            backend, max_workers = (window_parallel or (None, None))[0:2]

            if parallel and backend == "process":
                with _get_executor(
                        backend,
                        max_workers,
                        initializer=WorkerContext.init,
                        initargs=({"src_key": ("raster", in_path)},)
                ) as executor:
                    futures = [executor.submit(_align_process_window, *args) for args in window_args]
                    for args, future in zip(window_args, futures):
                        buf = future.result()
                        dst.write(buf, window=args[1])
                WorkerContext.close()
            else:
                WorkerContext.init({"src_key": ("raster", in_path)})
                for args in window_args:
                    buf = _align_process_window(*args)
                    dst.write(buf, window=args[1])
                WorkerContext.close()

def _align_process_window(
    src_win: Window,
    dst_win: Window,
    dst_transform,
    resampling_method: str,
    src_key: str,
    dst_shape: Tuple[int, int],
    debug_logs: bool = False,
    ) -> np.ndarray:

    src = WorkerContext.get(src_key)
    num_bands = src.count
    dst_buffer = np.empty((num_bands, dst_shape[0], dst_shape[1]), dtype=src.dtypes[0])

    reproject(
        source=rasterio.band(src, list(range(1, num_bands + 1))),
        destination=dst_buffer,
        src_transform=src.window_transform(src_win),
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=src.crs,
        src_nodata=src.nodata,
        dst_nodata=src.nodata,
        resampling=Resampling[resampling_method],
        src_window=src_win,
        dst_window=Window(0, 0, dst_shape[1], dst_shape[0]),
    )

    return dst_buffer


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
    input_image_paths: List[str],
    output_image_paths: List[str],
    vector_mask_path: str | None = None,
    split_mask_by_attribute: Optional[str] = None,
    resampling_method: Literal["nearest", "bilinear", "cubic"] = "nearest",
    tap: bool = False,
    resolution: Literal["highest", "average", "lowest"] = "highest",
    window_size: int | Tuple[int, int] | None = None,
    debug_logs: bool = False,
    include_touched_pixels: bool = False,
) -> None:
    """
    Masks rasters using vector geometries. If `split_mask_by_attribute` is set,
    geometries are filtered by the raster's basename (excluding extension) to allow
    per-image masking with specific matching features. If no vector is provided,
    rasters are processed without masking.

    Args:
        input_image_paths (List[str]): Paths to input rasters.
        output_image_paths (List[str]): Corresponding output raster paths.
        vector_mask_path (str | None, optional): Path to vector mask file (.shp, .gpkg, etc.). If None, no masking is applied.
        split_mask_by_attribute (Optional[str]): Attribute to match raster basenames.
        resampling_method (Literal["nearest", "bilinear", "cubic"]): Resampling algorithm.
        tap (bool): Snap output bounds to target-aligned pixels.
        resolution (Literal["highest", "average", "lowest"]): Strategy to determine target resolution.
        window_size (Optional[int | Tuple[int, int]]): Optional tile size for processing.
        debug_logs (bool): Print debug information if True.
        include_touched_pixels (bool): If True, include touched pixels in output raster.

    Outputs:
        Saved masked raster files to output_image_paths.
    """
    if debug_logs: print(f'Masking {len(input_image_paths)} rasters')

    gdf = gpd.read_file(vector_mask_path) if vector_mask_path else None

    if isinstance(window_size, int): window_size = (window_size, window_size)

    resolutions = []
    bounds_list = []
    crs_set = set()

    for path in input_image_paths:
        with rasterio.open(path) as src:
            res_x, res_y = src.res
            resolutions.append((res_x, res_y))
            bounds_list.append(src.bounds)
            crs_set.add(src.crs)

    if len(crs_set) > 1:
        raise ValueError("Input rasters must have the same CRS.")

    resolutions_array = np.array(resolutions)
    if resolution == "highest":
        target_res = resolutions_array.min(axis=0)
    elif resolution == "lowest":
        target_res = resolutions_array.max(axis=0)
    else:
        target_res = resolutions_array.mean(axis=0)
    if debug_logs: print(f'Resolution: {target_res}')

    temp_dir = tempfile.mkdtemp()
    tapped_paths = []

    for in_path in input_image_paths:
        raster_name = os.path.splitext(os.path.basename(in_path))[0]
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

                profile.update({
                    "height": dst_height,
                    "width": dst_width,
                    "transform": dst_transform
                })

                temp_path = os.path.join(temp_dir, f"{raster_name}_tapped.tif")
                tapped_paths.append(temp_path)

                src_windows = list(_create_windows(src.width, src.height, *window_size)) if window_size else [Window(0, 0, src.width, src.height)]
                dst_windows = list(_create_windows(dst_width, dst_height, *window_size)) if window_size else [Window(0, 0, dst_width, dst_height)]

                with rasterio.open(temp_path, "w", **profile) as dst:
                    for src_win, dst_win in zip(src_windows, dst_windows):
                        reproject(
                            source=rasterio.band(src, list(range(1, src.count + 1))),
                            destination=rasterio.band(dst, list(range(1, src.count + 1))),
                            src_transform=src.window_transform(src_win),
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=src.crs,
                            src_nodata=src.nodata,
                            dst_nodata=src.nodata,
                            resampling=Resampling[resampling_method],
                            src_window=src_win,
                            dst_window=dst_win
                        )
            else:
                tapped_paths.append(in_path)

    for in_path, out_path in zip(tapped_paths, output_image_paths):
        raster_name = os.path.splitext(os.path.basename(in_path))[0].replace('_tapped', '')
        if debug_logs: print(f'Processing: {raster_name}')
        with rasterio.open(in_path) as src:
            profile = src.profile.copy()
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            if window_size:
                windows = list(_create_windows(src.width, src.height, *window_size))
            else:
                windows = [Window(0, 0, src.width, src.height)]

            with rasterio.open(out_path, "w", **profile) as dst:
                for window in windows:
                    data = src.read(window=window)
                    if gdf is not None:
                        transform = src.window_transform(window)

                        if split_mask_by_attribute:
                            filtered_gdf = gdf[gdf[split_mask_by_attribute].str.strip() == raster_name.strip()]
                            if filtered_gdf.empty:
                                if debug_logs: print(f"No matching features for {raster_name}")
                                dst.write(data, window=window)
                                continue
                            geometries = filtered_gdf.geometry.values
                        else:
                            geometries = gdf.geometry.values

                        mask_array = geometry_mask(
                            geometries,
                            out_shape=(data.shape[1], data.shape[2]),
                            transform=transform,
                            invert=True,
                            all_touched=include_touched_pixels
                        )

                        data = np.where(mask_array, data, src.nodata)

                    dst.write(data, window=window)

    if tap:
        shutil.rmtree(temp_dir)


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