import os
import numpy as np
import tempfile
import rasterio
import shutil
import geopandas as gpd
import glob
import pandas as pd
import re

from typing import List, Optional, Literal, Tuple
from osgeo import ogr
from rasterio.windows import Window
from rasterio.warp import reproject
from rasterio.enums import Resampling
from .utils import _create_windows
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.coords import BoundingBox


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


def merge_rasters(
    input_images: Tuple[str, str] | List[str],
    output_image_path: str,
    window_size: Optional[int | Tuple[int, int]] = None,
    debug_logs: bool = False,
    output_dtype: str | None = None,
    custom_nodata_value: float | int | None = None,
    ) -> None:
    """
    Merges multiple input rasters into a single mosaic file by aligning each image geospatially and writing them in the correct location using tiling.

    Args:
        input_images (List[str]): Paths to input raster images.
        output_image_path (str): Path to save the merged output raster.
        window_size (int | Tuple[int, int] | None, optional): Tile size for memory-efficient processing.
        debug_logs (bool, optional): Enable debug logging.
        output_dtype (str | None, optional): Output dtype for output raster. None will default to input raster type.

    Output:
        A geospatially aligned, merged raster is saved to `output_image_path`.
    """
    if debug_logs: print('Start merging')

    if isinstance(input_images, tuple): input_images = search_paths(*input_images)

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    if isinstance(window_size, int):
        window_size = (window_size, window_size)

    # Read metadata and calculate combined bounds and resolution
    all_bounds = []
    all_res = []
    crs = None
    dtype = None
    count = None
    nodata_value = None

    for path in input_images:
        with rasterio.open(path) as src:
            all_bounds.append(src.bounds)
            all_res.append(src.res)
            if crs is None:
                crs = src.crs
                dtype = output_dtype or src.dtypes[0]
                count = src.count
                nodata_value = src.nodata

    minx = min(b.left for b in all_bounds)
    miny = min(b.bottom for b in all_bounds)
    maxx = max(b.right for b in all_bounds)
    maxy = max(b.top for b in all_bounds)

    res_x, res_y = all_res[0]  # Assume same resolution across rasters
    width = int(np.ceil((maxx - minx) / res_x))
    height = int(np.ceil((maxy - miny) / res_y))
    transform = from_origin(minx, maxy, res_x, res_y)

    if window_size:
        windows = list(_create_windows(width, height, *window_size))
    else:
        windows = [Window(0, 0, width, height)]

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata_value or None
    }
    # if debug_logs: print(f"xStart:yStart xSizeXySize ({len(windows)} windows): ", end="")
    with rasterio.open(output_image_path, 'w', **profile) as dst:
        for window in windows:
            win_transform = rasterio.windows.transform(window, transform)
            win_bounds = BoundingBox(*rasterio.windows.bounds(window, transform))
            merged_data = np.zeros((count, window.height, window.width), dtype=dtype)

            for path in input_images:
                with rasterio.open(path) as src:
                    src_bounds = src.bounds
                    if (
                            src_bounds.right <= win_bounds.left or
                            src_bounds.left >= win_bounds.right or
                            src_bounds.top <= win_bounds.bottom or
                            src_bounds.bottom >= win_bounds.top
                    ):
                        continue
                    try:
                        src_window = rasterio.windows.from_bounds(
                            *win_bounds,
                            transform=src.transform
                        )
                        src_data = src.read(
                            window=src_window,
                            out_shape=(count, window.height, window.width),
                            resampling=Resampling.nearest
                        )

                        nodata_val = src.nodata
                        if nodata_val is not None:
                            mask = ~(np.isclose(src_data, nodata_val))
                        else:
                            mask = (src_data != 0)

                        merged_data = np.where(mask, src_data, merged_data)

                    except Exception as e:
                        if debug_logs:
                            print(f"Skipping {path} in window {window} due to error: {e}")

            dst.write(merged_data, window=window)
            # if debug_logs: print(f"{window.col_off}:{window.row_off} {window.width}x{window.height}, ", end="", flush=True)
    if debug_logs: print("Done merging")


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
    create_folders: bool = False,
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