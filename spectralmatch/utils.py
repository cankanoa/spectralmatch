import os
import fiona
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np

from typing import Optional, Literal, Tuple
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject
from concurrent.futures import as_completed
from rasterio.transform import Affine

from .handlers import _resolve_paths, _check_raster_requirements
from .types_and_validation import Universal
from .utils_multiprocessing import (
    _resolve_windows,
    _get_executor,
    WorkerContext,
    _resolve_parallel_config,
)
from .handlers import _resolve_nodata_value


def merge_vectors(
    input_vectors: Universal.SearchFolderOrListFiles,
    merged_vector_path: str,
    method: Literal["intersection", "union", "keep"],
    debug_logs: bool = False,
    create_name_attribute: Optional[Tuple[str, str]] = None,
) -> None:
    """
    Merge multiple vector files using the specified geometric method.

    Args:
        input_vectors (str | List[str]): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.gpkg", "/input/folder" (assumes *.gpkg), ["/input/one.tif", "/input/two.tif"].
        merged_vector_path (str): Path to save merged output.
        method (Literal["intersection", "union", "keep"]): Merge strategy.
        debug_logs (bool): If True, print debug information.
        create_name_attribute (Optional[Tuple[str, str]]): Tuple of (field_name, separator) to add a combined name field.

    Returns:
        None
    """
    print("Start vector merge")

    os.makedirs(os.path.dirname(merged_vector_path), exist_ok=True)
    input_vector_paths = _resolve_paths(
        "search", input_vectors, kwargs={"default_file_pattern": "*.gpkg"}
    )

    geoms = []
    input_names = []

    for path in input_vector_paths:
        gdf = gpd.read_file(path)
        if create_name_attribute:
            name = os.path.splitext(os.path.basename(path))[0]
            input_names.append(name)
        geoms.append(gdf)

    combined_name_value = None
    if create_name_attribute:
        field_name, sep = create_name_attribute
        combined_name_value = sep.join(input_names)

    if method == "keep":
        merged_dfs = []
        field_name = create_name_attribute[0] if create_name_attribute else None
        for path in input_vector_paths:
            gdf = gpd.read_file(path)
            if field_name:
                name = os.path.splitext(os.path.basename(path))[0]
                gdf[field_name] = name
            merged_dfs.append(gdf)
        merged = gpd.GeoDataFrame(
            pd.concat(merged_dfs, ignore_index=True), crs=merged_dfs[0].crs
        )

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
    """
    Aligns multiple rasters to a common resolution and grid using specified resampling.

    Args:
        input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_images (str | List[str], required): Defines output files from a template path, folder, or list of paths (with the same length as the input). Specify like: "/input/files/$.tif", "/input/folder" (assumes $_Align.tif), ["/input/one.tif", "/input/two.tif"].
        resampling_method (Literal["nearest", "bilinear", "cubic"], optional): Resampling method to use; default is "bilinear".
        tap (bool, optional): If True, aligns outputs to target-aligned pixels (GDAL's -tap); default is False.
        resolution (Literal["highest", "average", "lowest"], optional): Strategy for choosing target resolution; default is "highest".
        window_size (Universal.WindowSize, optional): Tiling strategy for windowed alignment.
        debug_logs (Universal.DebugLogs, optional): If True, prints debug output.
        image_parallel_workers (Universal.ImageParallelWorkers, optional): Parallelization strategy for image-level alignment.
        window_parallel_workers (Universal.WindowParallelWorkers, optional): Parallelization strategy for within-image window alignment.

    Returns:
        None
    """

    print("Start align rasters")

    Universal.validate(
        input_images=input_images,
        output_images=output_images,
        debug_logs=debug_logs,
        window_size=window_size,
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
            "default_file_pattern": "$_Align.tif",
        },
    )
    image_names = _resolve_paths("name", input_image_paths)

    if debug_logs:
        print(f"{len(input_image_paths)} rasters to align")

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

    if debug_logs:
        print(f"Target resolution: {target_res}")

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
        for in_path, out_path, image_name in zip(
            input_image_paths, output_image_paths, image_names
        )
    ]

    if image_parallel_workers:
        with _get_executor(*image_parallel_workers) as executor:
            futures = [
                executor.submit(_align_process_image, *args) for args in parallel_args
            ]
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
    """
    Aligns a single raster image to a target resolution and grid, optionally in parallel by window.

    Args:
        image_name (str): Identifier for the image, used for worker context management.
        window_parallel (Universal.WindowParallelWorkers): Optional multiprocessing config for window-level alignment.
        in_path (str): Path to the input raster.
        out_path (str): Path to save the aligned output raster.
        target_res (Tuple[float, float]): Target resolution (x, y) to resample the raster to.
        resampling_method (str): Resampling method: "nearest", "bilinear", or "cubic".
        tap (bool): If True, aligns raster to target-aligned pixels (GDAL-style -tap).
        window_size (Universal.WindowSize): Tiling strategy for dividing the image into windows.
        debug_logs (bool): If True, prints debug output.

    Returns:
        None
    """

    if debug_logs:
        print(f"Aligning: {in_path}")

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

        profile.update(
            {"height": dst_height, "width": dst_width, "transform": dst_transform}
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            for band_idx in range(src.count):

                windows_dst = _resolve_windows(dst, window_size)

                window_args = []
                for dst_win in windows_dst:
                    dst_bounds = rasterio.windows.bounds(dst_win, dst.transform)

                    # Convert bounds to source window using inverse transform
                    src_win = rasterio.windows.from_bounds(
                        *dst_bounds, transform=src.transform
                    )
                    # src_win = src_win.round_offsets().round_lengths()  # Makes it integer-aligned

                    window_args.append(
                        (
                            src_win,
                            dst_win,
                            band_idx,
                            dst_transform,
                            resampling_method,
                            src.nodata,
                            debug_logs,
                            image_name,
                        )
                    )

                parallel = window_parallel is not None
                backend, max_workers = (window_parallel or (None, None))[0:2]

                if parallel and backend == "process":
                    with _get_executor(
                        backend,
                        max_workers,
                        initializer=WorkerContext.init,
                        initargs=({image_name: ("raster", in_path)},),
                    ) as executor:
                        futures = [
                            executor.submit(_align_process_window, *args)
                            for args in window_args
                        ]
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
    dst_transform_window = dst_transform * Affine.translation(
        dst_window.col_off, dst_window.row_off
    )

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
    input_images: Universal.SearchFolderOrListFiles,
    output_image_path: str,
    *,
    image_parallel_workers: Universal.ImageParallelWorkers = None,
    window_parallel_workers: Universal.WindowParallelWorkers = None,
    window_size: Universal.WindowSize = None,
    debug_logs: Universal.DebugLogs = False,
    output_dtype: Universal.CustomOutputDtype = None,
    custom_nodata_value: Universal.CustomNodataValue = None,
) -> None:
    """
    Merges multiple rasters into a single mosaic aligned to the union extent and minimum resolution.

    Args:
        input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_image_path (str): Path to save the merged output raster.
        image_parallel_workers (Universal.ImageParallelWorkers, optional): Strategy for parallelizing image-level merging.
        window_parallel_workers (Universal.WindowParallelWorkers, optional): Strategy for within-image window merging.
        window_size (Universal.WindowSize, optional): Tiling strategy for processing windows.
        debug_logs (Universal.DebugLogs, optional): If True, prints debug output.
        output_dtype (Universal.CustomOutputDtype, optional): Output data type; defaults to input type if None.
        custom_nodata_value (Universal.CustomNodataValue, optional): NoData value to use; defaults to first input's value.

    Returns:
        None
    """

    print("Start raster merging")

    # Validate parameters
    Universal.validate(
        input_images=input_images,
        debug_logs=debug_logs,
        custom_nodata_value=custom_nodata_value,
        output_dtype=output_dtype,
        window_size=window_size,
        image_parallel_workers=image_parallel_workers,
        window_parallel_workers=window_parallel_workers,
    )

    input_image_paths = _resolve_paths(
        "search", input_images, kwargs={"default_file_pattern": "*.tif"}
    )

    _check_raster_requirements(
        input_image_paths,
        debug_logs,
        check_geotransform=True,
        check_crs=True,
        check_bands=True,
        check_nodata=True,
    )

    image_names = [os.path.splitext(os.path.basename(p))[0] for p in input_image_paths]
    input_image_path_pairs = dict(zip(image_names, input_image_paths))

    if custom_nodata_value:
        nodata_value = custom_nodata_value
    else:
        with rasterio.open(input_image_paths[0]) as src:
            nodata_value = src.nodata

    if debug_logs:
        print(f"Merging {len(input_image_paths)} rasters into: {output_image_path}")

    # Compute union bounds and min resolution
    bounds_list = []
    res_x_list, res_y_list = [], []
    for path in input_image_paths:
        with rasterio.open(path) as src:
            bounds_list.append(src.bounds)
            res_x, res_y = src.res
            res_x_list.append(res_x)
            res_y_list.append(res_y)

    minx = min(b.left for b in bounds_list)
    miny = min(b.bottom for b in bounds_list)
    maxx = max(b.right for b in bounds_list)
    maxy = max(b.top for b in bounds_list)

    res_x = min(res_x_list)
    res_y = min(res_y_list)

    width = int(np.ceil((maxx - minx) / res_x))
    height = int(np.ceil((maxy - miny) / res_y))

    transform = Affine.translation(minx, maxy) * Affine.scale(res_x, -res_y)

    with rasterio.open(input_image_paths[0]) as src:
        meta = src.meta.copy()
        meta.update(
            {
                "height": height,
                "width": width,
                "transform": transform,
                "count": src.count,
                "dtype": output_dtype or src.dtypes[0],
                "nodata": nodata_value,
            }
        )

    # Determine multiprocessing and worker count
    image_parallel, image_backend, image_max_workers = _resolve_parallel_config(
        image_parallel_workers
    )

    parallel_args = []
    for name, path in input_image_path_pairs.items():
        with rasterio.open(path) as src:
            for band in range(src.count):
                windows = _resolve_windows(src, window_size)
                for window in windows:
                    parallel_args.append(
                        (
                            window,
                            band,
                            meta["dtype"],
                            debug_logs,
                            name,
                            src.transform,
                            transform,
                            nodata_value,
                        )
                    )

    # Pre-initialize WorkerContext
    init_worker = WorkerContext.init
    init_args_map = {
        name: ("raster", path) for name, path in input_image_path_pairs.items()
    }

    with rasterio.open(output_image_path, "w", **meta):
        pass
    with rasterio.open(output_image_path, "r+", **meta) as dst:
        if image_parallel:
            with _get_executor(
                image_backend,
                image_max_workers,
                initializer=init_worker,
                initargs=(init_args_map,),
            ) as executor:
                futures = [
                    executor.submit(_merge_raster_process_window, *args)
                    for args in parallel_args
                ]
                for future in as_completed(futures):
                    band, dst_window, buf = future.result()
                    if buf is not None:
                        existing = dst.read(band + 1, window=dst_window)
                        valid_mask = buf != nodata_value
                        merged = np.where(valid_mask, buf, existing)
                        dst.write(merged, band + 1, window=dst_window)
        else:
            WorkerContext.init(init_args_map)
            for args in parallel_args:
                band, dst_window, buf = _merge_raster_process_window(*args)
                if buf is not None:
                    existing = dst.read(band + 1, window=dst_window)
                    valid_mask = buf != nodata_value
                    merged = np.where(valid_mask, buf, existing)
                    dst.write(merged, band + 1, window=dst_window)
            WorkerContext.close()
    if debug_logs:
        print("Raster merging complete")


def _merge_raster_process_window(
    window: Window,
    band_idx: int,
    dtype: str,
    debug_logs: bool,
    image_name: str,
    src_transform,
    dst_transform,
    nodata_value: Universal.CustomNodataValue,
) -> tuple[int, Window, np.ndarray]:
    """
    Processes a single raster window for merging by reading, masking, and mapping it to the destination grid.

    Args:
        window (Window): Source window to read.
        band_idx (int): Zero-based band index to process.
        dtype (str): Data type to cast the read block to.
        debug_logs (bool): If True, prints debug output.
        image_name (str): Identifier for accessing the source dataset from WorkerContext.
        src_transform: Affine transform of the source image.
        dst_transform: Affine transform of the destination mosaic.
        nodata_value (Universal.CustomNodataValue): Value representing NoData pixels.

    Returns:
        tuple[int, Window, np.ndarray]: Band index, destination window, and processed data block (or None if fully masked).
    """

    # Read the block from the source image
    ds = WorkerContext.get(image_name)
    block = ds.read(band_idx + 1, window=window).astype(dtype)

    if nodata_value is not None:
        mask = block == nodata_value
        if mask.all():
            return band_idx, None, None
        block[mask] = nodata_value

    row_off, col_off = int(window.row_off), int(window.col_off)
    x, y = src_transform * (col_off, row_off)

    # Convert world coordinates to destination pixel space
    dst_col_off, dst_row_off = ~dst_transform * (x, y)

    # Reuse original width/height
    dst_window = Window(
        col_off=int(round(dst_col_off)),
        row_off=int(round(dst_row_off)),
        width=window.width,
        height=window.height,
    )

    return band_idx, dst_window, block


def mask_rasters(
    input_images: Universal.SearchFolderOrListFiles,
    output_images: Universal.CreateInFolderOrListFiles,
    vector_mask: Universal.VectorMask = None,
    window_size: Universal.WindowSize = None,
    debug_logs: Universal.DebugLogs = False,
    image_parallel_workers: Universal.ImageParallelWorkers = None,
    window_parallel_workers: Universal.WindowParallelWorkers = None,
    include_touched_pixels: bool = False,
    custom_nodata_value: Universal.CustomNodataValue = None,
) -> None:
    """
    Applies a vector-based mask to one or more rasters, with support for image- and window-level parallelism.

    Args:
        input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_images (str | List[str], required): Defines output files from a template path, folder, or list of paths (with the same length as the input). Specify like: "/input/files/$.tif", "/input/folder" (assumes $_Clip.tif), ["/input/one.tif", "/input/two.tif"].
        vector_mask (Universal.VectorMask, optional): Tuple ("include"/"exclude", vector path, optional field name) or None.
        window_size (Universal.WindowSize, optional): Strategy for tiling rasters during processing.
        debug_logs (Universal.DebugLogs, optional): If True, prints debug information.
        image_parallel_workers (Universal.ImageParallelWorkers, optional): Strategy for parallelizing image-level masking.
        window_parallel_workers (Universal.WindowParallelWorkers, optional): Strategy for parallelizing masking within windows.
        include_touched_pixels (bool, optional): If True, includes pixels touched by mask geometry edges; default is False.

    Returns:
        None
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
        custom_nodata_value=custom_nodata_value,
    )

    input_image_paths = _resolve_paths(
        "search", input_images, kwargs={"default_file_pattern": "*.tif"}
    )
    output_image_paths = _resolve_paths(
        "create",
        output_images,
        kwargs={
            "paths_or_bases": input_image_paths,
            "default_file_pattern": "$_Clip.tif",
        },
    )

    if debug_logs:
        print(f"Input images: {input_image_paths}")
    if debug_logs:
        print(f"Output images: {output_image_paths}")

    input_image_names = [
        os.path.splitext(os.path.basename(p))[0] for p in input_image_paths
    ]
    input_image_path_pairs = dict(zip(input_image_names, input_image_paths))
    output_image_path_pairs = dict(zip(input_image_names, output_image_paths))

    image_parallel, image_backend, image_max_workers = _resolve_parallel_config(
        image_parallel_workers
    )
    window_parallel, window_backend, window_max_workers = _resolve_parallel_config(
        window_parallel_workers
    )

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
            include_touched_pixels,
            custom_nodata_value,
        )
        for name in input_image_names
    ]

    if image_parallel:
        with _get_executor(image_backend, image_max_workers) as executor:
            futures = [
                executor.submit(_mask_raster_process_image, *args)
                for args in parallel_args
            ]
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
    custom_nodata_value: Universal.CustomNodataValue,
):
    """
    Processes a single raster image by applying a vector mask, optionally in parallel by window.

    Args:
        window_parallel (bool): Whether to use parallel processing at the window level.
        max_workers (int): Maximum number of worker processes or threads.
        backend (str): Execution backend, e.g., "process".
        input_image_path (str): Path to the input raster.
        output_image_path (str): Path to save the masked output raster.
        image_name (str): Identifier for the raster used in worker context.
        vector_mask (Universal.VectorMask): Masking config as ("include"/"exclude", path, optional field).
        window_size (Universal.WindowSize): Strategy for tiling the raster into windows.
        debug_logs (bool): If True, enables debug output.
        include_touched_pixels (bool): If True, includes pixels touched by mask geometry boundaries.

    Returns:
        None
    """

    with rasterio.open(input_image_path) as src:
        profile = src.profile.copy()
        nodata_val = _resolve_nodata_value(src, custom_nodata_value)
        assert (
            nodata_val is not None
        ), "Nodata value must be set via custom_nodata_value or in the raster metadata."

        profile["nodata"] = nodata_val
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
                        if field_name in feat["properties"]
                        and image_name in str(feat["properties"][field_name])
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
                        include_touched_pixels,
                    )
                    for win in windows
                ]

                if window_parallel:
                    with _get_executor(
                        backend,
                        max_workers,
                        initializer=WorkerContext.init,
                        initargs=({image_name: ("raster", input_image_path)},),
                    ) as executor:
                        futures = [
                            executor.submit(_mask_raster_process_window, *arg)
                            for arg in args
                        ]
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
    """
    Applies a vector-based mask to a single raster window and returns the masked data.

    Args:
        win (Window): Raster window to process.
        band_idx (int): Zero-based band index to read.
        image_name (str): Identifier for the raster in the WorkerContext.
        nodata (int | float): Value to assign to masked-out pixels.
        geoms (list | None): List of geometries to mask with, or None to skip masking.
        invert (bool): If True, masks outside the geometries (exclude mode).
        include_touched_pixels (bool): If True, includes pixels touched by mask boundaries.

    Returns:
        tuple[Window, np.ndarray]: The window and its corresponding masked data array.
    """
    src = WorkerContext.get(image_name)
    mask_key = (
        f"{int(win.col_off)}-{int(win.row_off)}-{int(win.width)}-{int(win.height)}"
    )

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
                all_touched=include_touched_pixels,
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
