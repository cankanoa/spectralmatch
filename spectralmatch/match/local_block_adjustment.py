import multiprocessing as mp
import math
import os
import numpy as np
import rasterio
import traceback
import gc
import fiona

from scipy.ndimage import map_coordinates, gaussian_filter
from rasterio.windows import from_bounds, Window
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Optional, List, Literal, Union
from multiprocessing import Lock
from multiprocessing import shared_memory

from ..utils import _check_raster_requirements, _get_nodata_value
from ..handlers import create_paths, search_paths, match_paths
from ..utils_multiprocessing import _create_windows, _choose_context, _resolve_parallel_config, _resolve_windows, _get_executor, WorkerContext


# Multiprocessing setup
_worker_dataset_cache = {}
file_lock = Lock()


def local_block_adjustment(
    input_images: Tuple[str, str] | List[str],
    output_images: Tuple[str, str] | List[str],
    *,
    custom_nodata_value: float | int | None = None,
    number_of_blocks: int | Tuple[int, int] | Literal["coefficient_of_variation"] = 100,
    alpha: float = 1.0,
    calculation_dtype: str = "float32",
    output_dtype: str | None = None,
    debug_logs: bool = False,
    window_size: int | Tuple[int, int] | Literal["block"] | None = None,
    correction_method: Literal["gamma", "linear"] = "gamma",
    image_parallel_workers: Tuple[Literal["process", "thread"], Literal["cpu"] | int] | None = None,
    window_parallel_workers: Tuple[Literal["process", "thread"], Literal["cpu"] | int] | None = None,
    save_block_maps: Tuple[str, str] | None = None,
    load_block_maps: Tuple[str, List[str]] | Tuple[str, None]| Tuple[None, List[str]] | None = None,
    override_bounds_canvas_coords: Tuple[float, float, float, float] | None = None,
    vector_mask_path: Tuple[Literal["include", "exclude"], str] | Tuple[Literal["include", "exclude"], str, str] | None = None,
    block_valid_pixel_threshold: float = 0.001,
    )-> list:
    """
    Performs local radiometric adjustment on a set of raster images using block-based statistics.

    Args:
        input_images (Tuple[str, str] | List[str]):
            Specifies the input images either as:
            - A tuple with a folder path and glob pattern to search for files (e.g., ("/input/folder", "*.tif")).
            - A list of full file paths to individual input images.
        output_images (Tuple[str, str] | List[str]):
            Specifies how output filenames are generated or provided:
            - A tuple with an output folder and a filename template using "$" as a placeholder for each input image's basename (e.g., ("/output/folder", "$_LocalMatch.tif")).
            - A list of full output paths, which must match the number of input images.
        custom_nodata_value (float | int | None, optional): Overrides detected NoData value. Defaults to None.
        number_of_blocks (int | tuple | Literal["coefficient_of_variation"]): int as a target of blocks per image, tuple to set manually set total blocks width and height, coefficient_of_variation to find the number of blocks based on this metric.
        alpha (float, optional): Blending factor between reference and local means. Defaults to 1.0.
        calculation_dtype (str, optional): Precision for internal calculations. Defaults to "float32".
        output_dtype (str | None, optional): Data type for output rasters. Defaults to input image dtype.
        debug_logs (bool, optional): If True, prints progress. Defaults to False.
        window_size (int | Tuple[int, int] | Literal["block"] | None): Tile size for processing: int for square tiles, (width, height) for custom size, or "block" to set as the size of the block map, None for full image. Defaults to None.
        correction_method (Literal["gamma", "linear"], optional): Local correction method. Defaults to "gamma".
        image_parallel_workers (Tuple[Literal["process", "thread"], Literal["cpu"] | int] | None = None): Parallelization strategy at the image level. Provide a tuple like ("process", "cpu") to use multiprocessing with all available cores, or ("thread", 4) to use 4 threads. Set to None to disable.
        window_parallel_workers (Tuple[Literal["process", "thread"], Literal["cpu"] | int] | None = None): Parallelization strategy at the window level within each image. Same format as image_parallel_workers. Enables finer-grained parallelism across image tiles. Set to None to disable.
        save_block_maps (tuple(str, str) | None): If enabled, saves block maps for review, to resume processing later, or to add additional images to the reference map.
            - First str is the path to save the global block map.
            - Second str is the path to save the local block maps, which must include "$" which will be replaced my the image name (because there are multiple local maps).
        load_block_maps (Tuple[str, List[str]] | Tuple[str, None] | Tuple[None, List[str]] | None, optional):
            Controls loading of precomputed block maps. Can be one of:
                - Tuple[str, List[str]]: Load both reference and local block maps.
                - Tuple[str, None]: Load only the reference block map.
                - Tuple[None, List[str]]: Load only the local block maps.
                - None: Do not load any block maps.
            This supports partial or full reuse of precomputed block maps:
                - Local block maps will still be computed for each input image that is not linked to a local block map by the images name being *included* in the local block maps name (file name).
                - The reference block map will only be calculated (mean of all local blocks) if not set.
                - The reference map defines the reference block statistics and the local maps define per-image local block statistics.
                - Both reference and local maps must have the same canvas extent and dimensions which will be used to set those values.
        override_bounds_canvas_coords (Tuple[float, float, float, float] | None): Manually set (min_x, min_y, max_x, max_y) bounds to override the computed/loaded canvas extent. If you wish to have a larger extent than the current images, you can manually set this, along with setting a fixed number of blocks, to anticipate images will expand beyond the current extent.
        vector_mask_path (Tuple[Literal["include", "exclude"], str] | Tuple[Literal["include", "exclude"], str, str] | None): A mask limiting pixels to include when calculating stats for each block in the format of a tuple with two or three items: literal "include" or "exclude" the mask area, str path to the vector file, optional str of field name in vector file that *includes* (can be substring) input image name to filter geometry by. It is only applied when calculating local blocks, as the reference map is calculated as the mean of all local blocks. Loaded block maps won't have this applied unless it was used when calculating them. The matching solution is still applied to these areas in the output. Defaults to None for no mask.
        block_valid_pixel_threshold (float): Minimum fraction of valid pixels required to include a block (0–1).

    Returns:
        List[str]: Paths to the locally adjusted output raster images.
    """

    print("Start local block adjustment")

    _validate_input_params(
        input_images,
        output_images,
        custom_nodata_value,
        number_of_blocks,
        alpha,
        calculation_dtype,
        output_dtype,
        debug_logs,
        window_size,
        correction_method,
        parallel_workers,
        save_block_maps,
        load_block_maps,
        override_bounds_canvas_coords,
        vector_mask_path,
        block_valid_pixel_threshold,
    )

    # Determine multiprocessing and worker count
    image_parallel, image_backend, image_max_workers = _resolve_parallel_config(image_parallel_workers)
    window_parallel, window_backend, window_max_workers = _resolve_parallel_config(window_parallel_workers)

    if isinstance(input_images, tuple): input_images = search_paths(*input_images)
    if isinstance(output_images, tuple): output_images = create_paths(*output_images, input_images, create_folders=True)

    if debug_logs: print(f"Input images: {input_images}")
    if debug_logs: print(f"Output images: {output_images}")

    input_image_paths = input_images
    input_image_names = [os.path.splitext(os.path.basename(p))[0] for p in input_images]
    input_image_pairs = dict(zip(input_image_names, input_images))
    output_image_pairs = dict(zip(input_image_names, output_images))

    _check_raster_requirements(input_image_paths, debug_logs)

    if isinstance(window_size, int): window_size = (window_size, window_size)
    nodata_val = _get_nodata_value(input_image_paths, custom_nodata_value)
    projection = rasterio.open(input_image_paths[0]).crs
    if debug_logs: print(f"Global nodata value: {nodata_val}")
    with rasterio.open(input_image_paths[0]) as ds:num_bands = ds.count

    # Load data from precomputed block maps if set
    if load_block_maps:
        loaded_block_local_means, loaded_block_reference_mean, loaded_num_row, loaded_num_col, loaded_bounds_canvas_coords = _get_pre_computed_block_maps(load_block_maps, calculation_dtype, debug_logs)
        loaded_names = list(loaded_block_local_means.keys())
        block_reference_mean = loaded_block_reference_mean

        matched = list((soft_matches := {
            input_name: loaded_name
            for input_name in input_image_names
            for loaded_name in loaded_names
            if input_name in loaded_name
        }).keys())
        only_loaded = [l for l in loaded_names if not any(n in l for n in input_image_names)]
        only_input = [n for n in input_image_names if not any(n in l for l in loaded_names)]

    else:
        only_input = input_image_names
        matched = []
        only_loaded = []
        block_reference_mean = None

    if debug_logs:
        print(f"Total images: input images: {len(input_image_names)}, loaded local block maps: {len(loaded_names) if load_block_maps else 0}:")
        print(f"    Matched local block maps (to override) ({len(matched)}):", sorted(matched))
        print(f"    Only in loaded local block maps (to use) ({len(only_loaded)}):", sorted(only_loaded))
        print(f"    Only in input (to compute) ({len(only_input)}):", sorted(only_input))

    # Unpack path to save block maps
    if save_block_maps:
        reference_map_path, local_map_path = save_block_maps

    # Create image bounds dict
    bounds_images_coords = {
        name: rasterio.open(path).bounds
        for name, path in input_image_pairs.items()
    }

    # Get bounds canvas coords
    if not override_bounds_canvas_coords:
        if not load_block_maps:
            bounds_canvas_coords = _get_bounding_rectangle(input_image_paths)
        else:
            bounds_canvas_coords = loaded_bounds_canvas_coords
    else:
        bounds_canvas_coords = override_bounds_canvas_coords
        if load_block_maps:
            if bounds_canvas_coords != loaded_bounds_canvas_coords:
                raise ValueError("Override bounds canvas coordinates do not match loaded block maps bounds")

    # Calculate the number of blocks
    if not load_block_maps:
        if isinstance(number_of_blocks, int):
            num_row, num_col = _compute_block_size(input_image_paths, number_of_blocks, bounds_canvas_coords)
        elif isinstance(number_of_blocks, tuple):
            num_row, num_col = number_of_blocks
        elif isinstance(number_of_blocks, str):
            num_row, num_col = _compute_mosaic_coefficient_of_variation(input_image_paths, nodata_val) # This is the approach from the paper to compute bock size
    else:
        num_row, num_col = loaded_num_row, loaded_num_col

    if debug_logs: print("Computing local block maps:")

    # Compute local blocks
    local_blocks_to_calculate = {k: v for k, v in input_image_pairs.items() if k in only_input}
    local_blocks_to_load = {
        **{k: loaded_block_local_means[soft_matches[k]] for k in matched},
        **{k: loaded_block_local_means[k] for k in only_loaded},
    }

    if local_blocks_to_calculate:
        block_local_means, block_local_counts = _compute_local_blocks(
            local_blocks_to_calculate,
            bounds_canvas_coords,
            num_row,
            num_col,
            num_bands,
            window_size,
            debug_logs,
            nodata_val,
            calculation_dtype,
            vector_mask_path,
            block_valid_pixel_threshold,
        )
        overlap = set(block_local_means) & set(local_blocks_to_load)
        if overlap: raise ValueError(f"Duplicate keys when merging loaded and computed blocks: {overlap}")

        block_local_means = {**block_local_means, **local_blocks_to_load}
    else:
        block_local_means = local_blocks_to_load


    bounds_images_block_space = get_bounding_rect_images_block_space(block_local_means)

    # Compute reference block
    if debug_logs: print("Computing reference block map")
    if block_reference_mean is None:
        block_reference_mean = _compute_reference_blocks(
            block_local_means,
            calculation_dtype,
            )

    if save_block_maps:
        _download_block_map(
            np.nan_to_num(block_reference_mean, nan=nodata_val),
            bounds_canvas_coords,
            reference_map_path,
            projection,
            calculation_dtype,
            nodata_val,
            num_col,
            num_row,
        )
        for name, block_local_mean in block_local_means.items():
            _download_block_map(
                np.nan_to_num(block_local_mean, nan=nodata_val),
                bounds_canvas_coords,
                local_map_path.replace("$", name),
                projection,
                calculation_dtype,
                nodata_val,
                num_col,
                num_row,
            )
            # _download_block_map(
            #     np.nan_to_num(block_local_count, nan=nodata_val),
            #     bounds_canvas_coords,
            #     os.path.join(output_image_folder, "BlockLocalCount", f"{input_image_name}_BlockLocalCount.tif"),
            #     projection,
            #     calculation_dtype,
            #     nodata_val,
            #     num_col,
            #     num_row,
            # )

    # block_local_mean = _smooth_array(block_local_mean, nodata_value=global_nodata_value)

    if debug_logs: print(f"Computing local correction, applying, and saving:")
    out_paths: List[str] = []
    for name, img_path in input_image_pairs.items():
        in_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = output_image_pairs[name]
        out_name = os.path.splitext(os.path.basename(out_path))[0]
        out_paths.append(str(out_path))

        if debug_logs: print(f"    {in_name}")
        with rasterio.open(img_path) as src:
            meta = src.meta.copy()
            meta.update({"count": num_bands, "dtype": output_dtype or src.dtypes[0], "nodata": nodata_val})
            block_reference_mean_masked = np.where(
                (np.arange(block_reference_mean.shape[0])[:, None, None] >= bounds_images_block_space[name][0]) &
                (np.arange(block_reference_mean.shape[0])[:, None, None] < bounds_images_block_space[name][2]) &
                (np.arange(block_reference_mean.shape[1])[None, :, None] >= bounds_images_block_space[name][1]) &
                (np.arange(block_reference_mean.shape[1])[None, :, None] < bounds_images_block_space[name][3]),
                block_reference_mean,
                np.nan
            )

            if isinstance(window_size, tuple):
                tw, th = window_size
                windows = list(_create_windows(src.width, src.height, tw, th))
            elif window_size == "block":
                block_width_geo = (bounds_canvas_coords[2] - bounds_canvas_coords[0]) / num_col
                block_height_geo = (bounds_canvas_coords[3] - bounds_canvas_coords[1]) / num_row
                res_x = abs(src.transform.a)
                res_y = abs(src.transform.e)
                tile_width = max(1, int(round(block_width_geo / res_x)))
                tile_height = max(1, int(round(block_height_geo / res_y)))
                windows = list(_create_windows(src.width, src.height, tile_width, tile_height))
            elif window_size is None:
                windows = [Window(0, 0, src.width, src.height)]

            if parallel:
                ctx = _choose_context(prefer_fork=True)

                ref_shm = shared_memory.SharedMemory(create=True, size=block_reference_mean.nbytes)
                ref_array = np.ndarray(block_reference_mean.shape, dtype=block_reference_mean.dtype, buffer=ref_shm.buf)
                ref_array[:] = block_reference_mean[:]

                loc_shm = shared_memory.SharedMemory(create=True, size=block_local_means[name].nbytes)
                loc_array = np.ndarray(block_local_means[name].shape, dtype=block_local_means[name].dtype,
                                       buffer=loc_shm.buf)
                loc_array[:] = block_local_means[name][:]

                pool = ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=ctx,
                    initializer=_init_worker,
                    initargs=(
                    img_path, ref_shm.name, loc_shm.name, block_reference_mean.shape, block_local_means[name].shape,
                    block_reference_mean.dtype.name),
                )

                try:
                    with rasterio.open(out_path, "w", **meta) as dst:
                        futures = [
                            pool.submit(_compute_tile_local,
                                        w,
                                        b,
                                        num_row,
                                        num_col,
                                        bounds_canvas_coords,
                                        nodata_val,
                                        alpha,
                                        correction_method,
                                        calculation_dtype,
                                        )
                            for b in range(num_bands)
                            for w_id, w in enumerate(windows)
                        ]
                        for fut in as_completed(futures):
                            win, b_idx, buf = fut.result()
                            dst.write(np.nan_to_num(buf, nan=nodata_val).astype(output_dtype), b_idx + 1, window=win)
                            del buf, win
                finally:
                    pool.shutdown(wait=True)
                    ref_shm.close()
                    loc_shm.close()
                    ref_shm.unlink()
                    loc_shm.unlink()
            else:
                with rasterio.open(out_path, "w", **meta) as dst:
                    _worker_dataset_cache["ds"] = rasterio.open(img_path, "r")
                    _worker_dataset_cache["block_ref_mean"] = block_reference_mean
                    _worker_dataset_cache["block_loc_mean"] = block_local_means[name]

                    for b in range(num_bands):
                        for w_id, win in enumerate(windows):
                            win_, b_idx, buf = _compute_tile_local(
                                win,
                                b,
                                num_row,
                                num_col,
                                bounds_canvas_coords,
                                nodata_val,
                                alpha,
                                correction_method,
                                calculation_dtype,
                            )
                            dst.write(buf.astype(output_dtype), b_idx + 1, window=win_)
                            del buf, win_
            if not parallel:
                if "ds" in _worker_dataset_cache:
                    _worker_dataset_cache["ds"].close()
                    del _worker_dataset_cache["ds"]
                _worker_dataset_cache.pop("block_ref_mean", None)
                _worker_dataset_cache.pop("block_loc_mean", None)

            del block_reference_mean_masked, windows
            gc.collect()
    return out_paths


def _validate_input_params(
    input_images,
    output_images,
    custom_nodata_value,
    number_of_blocks,
    alpha,
    calculation_dtype,
    output_dtype,
    debug_logs,
    window_size,
    correction_method,
    parallel_workers,
    save_block_maps,
    load_block_maps,
    override_bounds_canvas_coords,
    vector_mask_path,
    block_valid_pixel_threshold,
):
    """
    Validates input parameters for `local_block_adjustment`.

    Raises:
        TypeError or ValueError with a concise message if any parameter is improperly set.
    """
    if not (
        isinstance(input_images, tuple) and len(input_images) == 2 and all(isinstance(p, str) for p in input_images)
        or isinstance(input_images, list) and all(isinstance(p, str) for p in input_images)
    ):
        raise TypeError("input_images must be a tuple (folder, pattern) or a list of file paths.")

    if not (
        isinstance(output_images, tuple) and len(output_images) == 2 and all(isinstance(p, str) for p in output_images)
        or isinstance(output_images, list) and all(isinstance(p, str) for p in output_images)
    ):
        raise TypeError("output_images must be a tuple (folder, template) or a list of file paths.")

    if isinstance(output_images, tuple) and "$" not in output_images[1]:
        raise ValueError("The output filename template must include a '$' placeholder to insert the image name.")

    if custom_nodata_value is not None and not isinstance(custom_nodata_value, (int, float)):
        raise TypeError("custom_nodata_value must be a number or None.")

    if not (
        isinstance(number_of_blocks, int) or
        (isinstance(number_of_blocks, tuple) and len(number_of_blocks) == 2 and all(isinstance(n, int) for n in number_of_blocks)) or
        number_of_blocks == "coefficient_of_variation"
    ):
        raise ValueError("number_of_blocks must be an int, (int, int), or 'coefficient_of_variation'.")

    if not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a number.")

    if not isinstance(calculation_dtype, str):
        raise TypeError("calculation_dtype must be a string.")

    if output_dtype is not None and not isinstance(output_dtype, str):
        raise TypeError("output_dtype must be a string or None.")

    if not isinstance(debug_logs, bool):
        raise TypeError("debug_logs must be a boolean.")

    if not (
        window_size is None or
        isinstance(window_size, int) or
        (isinstance(window_size, tuple) and len(window_size) == 2 and all(isinstance(w, int) for w in window_size)) or
        window_size == "block"
    ):
        raise ValueError("window_size must be int, (int, int), 'block', or None.")

    if correction_method not in {"gamma", "linear"}:
        raise ValueError("correction_method must be 'gamma' or 'linear'.")

    if not (
        parallel_workers is None or
        parallel_workers == "cpu" or
        isinstance(parallel_workers, int)
    ):
        raise ValueError("parallel_workers must be None, 'cpu', or an integer.")

    if save_block_maps is not None:
        if not (isinstance(save_block_maps, tuple) and len(save_block_maps) == 2 and all(isinstance(p, str) for p in save_block_maps)):
            raise TypeError("save_block_maps must be a tuple of two strings or None.")
        if "$" not in save_block_maps[1]:
            raise ValueError("The local block map path template in save_block_maps must contain a '$' placeholder.")

    if load_block_maps is not None:
        if not (
            isinstance(load_block_maps, tuple) and len(load_block_maps) == 2 and
            (isinstance(load_block_maps[0], str) or load_block_maps[0] is None) and
            (isinstance(load_block_maps[1], list) or load_block_maps[1] is None)
        ):
            raise TypeError("load_block_maps must be (str, list), (str, None), (None, list), or None.")
        if isinstance(load_block_maps[1], list) and not all(isinstance(p, str) for p in load_block_maps[1]):
            raise TypeError("All elements in the local block maps list must be strings.")

    if override_bounds_canvas_coords is not None:
        if not (
            isinstance(override_bounds_canvas_coords, tuple) and
            len(override_bounds_canvas_coords) == 4 and
            all(isinstance(v, (int, float)) for v in override_bounds_canvas_coords)
        ):
            raise TypeError("override_bounds_canvas_coords must be a tuple of four numbers or None.")

    if vector_mask_path is not None:
        if not (
            isinstance(vector_mask_path, tuple) and
            len(vector_mask_path) in {2, 3} and
            vector_mask_path[0] in {"include", "exclude"} and
            isinstance(vector_mask_path[1], str) and
            (len(vector_mask_path) == 2 or isinstance(vector_mask_path[2], str))
        ):
            raise TypeError("vector_mask_path must be a tuple ('include'|'exclude', path [, field_name]) or None.")

    if not isinstance(block_valid_pixel_threshold, float) or not (0 <= block_valid_pixel_threshold <= 1):
        raise ValueError("block_valid_pixel_threshold must be a float between 0 and 1.")

def _get_pre_computed_block_maps(
    load_block_maps: Tuple[Optional[str], Optional[List[str]]],
    calculation_dtype: str,
    debug_logs: bool,
) -> Tuple[dict[str, np.ndarray], Optional[np.ndarray], Optional[int], Optional[int], Optional[Tuple[float, float, float, float]]]:
    """
    Load pre-computed block mean maps from files.

    Args:
        load_block_maps (Tuple[str, List[str]] | Tuple[str, None] | Tuple[None, List[str]]):
            - Tuple[str, List[str]]: Load both reference and local block maps.
            - Tuple[str, None]: Load only the reference block map.
            - Tuple[None, List[str]]: Load only the local block maps.
        calculation_dtype (str): Numpy dtype to use for reading.
        debug_logs (bool): To print debug statements or not.

    Returns:
        Tuple[
            dict[str, np.ndarray],             # block_local_means
            Optional[np.ndarray],              # block_reference_mean
            Optional[int],                     # num_row
            Optional[int],                     # num_col
            Optional[Tuple[float, float, float, float]]  # bounds_canvas_coords
        ]
    """
    ref_path, local_paths = load_block_maps

    shapes = set()
    extents = set()

    block_reference_mean = None

    # Load reference block map if provided
    if ref_path is not None:
        with rasterio.open(ref_path) as src:
            ref_data = src.read().astype(calculation_dtype).transpose(1, 2, 0)
            nodata_val = src.nodata
            if nodata_val is not None:
                ref_data[ref_data == nodata_val] = np.nan
            block_reference_mean = ref_data
            shapes.add(ref_data.shape)
            extents.add(src.bounds)

    # Load local block maps if provided
    block_local_means = {}
    if local_paths is not None:
        for p in local_paths:
            name = os.path.splitext(os.path.basename(p))[0]
            with rasterio.open(p) as src:
                data = src.read().astype(calculation_dtype).transpose(1, 2, 0)
                nodata_val = src.nodata
                if nodata_val is not None:
                    data[data == nodata_val] = np.nan

                shapes.add(data.shape)
                extents.add(src.bounds)
                block_local_means[name] = data

    if not shapes:
        raise ValueError("No block maps provided.")

    if len(shapes) != 1:
        raise ValueError(f"Inconsistent block map shapes: {shapes}")
    if len(extents) != 1:
        raise ValueError(f"Inconsistent block map extents: {extents}")

    num_row, num_col, _ = shapes.pop()
    bounds_canvas_coords = extents.pop()

    if debug_logs:
        print(f"Loaded block maps consistently have shape {(num_row, num_col)} and extent {bounds_canvas_coords}")

    return block_local_means, block_reference_mean, num_row, num_col, bounds_canvas_coords


def get_bounding_rect_images_block_space(
    block_local_means: dict[str, np.ndarray]
) -> dict[str, tuple[int, int, int, int]]:
    """
    Compute block-space bounding rectangles for each image based on valid block values.

    Args:
        block_local_means (dict[str, np.ndarray]): Per-image block means
            with shape (num_row, num_col, num_bands).

    Returns:
        dict[str, tuple[int, int, int, int]]: Each entry maps image name to
            (min_row, min_col, max_row, max_col).
    """
    output = {}

    for name, arr in block_local_means.items():
        valid_mask = np.any(~np.isnan(arr), axis=2)
        rows, cols = np.where(valid_mask)

        if rows.size > 0 and cols.size > 0:
            min_row, max_row = rows.min(), rows.max() + 1
            min_col, max_col = cols.min(), cols.max() + 1
        else:
            min_row = max_row = min_col = max_col = 0

        output[name] = (min_row, min_col, max_row, max_col)

    return output


def _compute_reference_blocks(
    block_local_means: dict[str, np.ndarray],
    calculation_dtype: str,
) -> np.ndarray:
    """
    Computes reference block means across images by averaging non-NaN local block means.

    Args:
        block_local_means (dict[str, np.ndarray]): Per-image block mean arrays.
        calculation_dtype (str): Numpy dtype for output array.

    Returns:
        np.ndarray: Reference block map of shape (num_row, num_col, num_bands)
    """
    shape = next(iter(block_local_means.values())).shape
    stacked = np.stack(list(block_local_means.values()), axis=0)  # shape: (num_images, H, W, B)
    with np.errstate(invalid='ignore'):
        valid_mask = np.any(~np.isnan(stacked), axis=0)
        ref_block_mean = np.full(shape, np.nan, dtype=calculation_dtype)
        ref_block_mean[valid_mask] = np.nanmean(stacked[:, valid_mask], axis=0).astype(calculation_dtype)
    return ref_block_mean


def _compute_tile_local(
    window: Window,
    band_idx: int,
    num_row: int,
    num_col: int,
    bounds_canvas_coords: tuple,
    nodata_val: float | int,
    alpha: float,
    correction_method: Literal["gamma", "linear"],
    calculation_dtype: str,
    ):
    """
    Applies local radiometric correction to a raster tile using bilinear interpolation between global and local block means.

    Args:
        window (Window): Rasterio window defining the tile extent.
        band_idx (int): Index of the band to process.
        num_row (int): Number of block rows in the mosaic.
        num_col (int): Number of block columns in the mosaic.
        bounds_canvas_coords (tuple): (minx, miny, maxx, maxy) bounding the full mosaic.
        nodata_val (float | int): NoData value in the raster.
        alpha (float): Weighting factor for blending reference and local statistics.
        correction_method (Literal["gamma", "linear"]): Method of radiometric correction.
        calculation_dtype (str): Internal computation precision (e.g., "float32").

    Returns:
        tuple: (Window, band index, corrected tile as np.ndarray)
    """
    try:
        # if debug_logs: print(f"b{band_idx}w{w_id}[{window.col_off}:{window.row_off} {window.width}x{window.height}], ", end="", flush=True)

        ds = _worker_dataset_cache["ds"]
        arr_in = ds.read(band_idx + 1, window=window).astype(calculation_dtype)
        arr_out = np.full_like(arr_in, nodata_val, dtype=calculation_dtype)

        mask = arr_in != nodata_val
        if not np.any(mask):
            return window, band_idx, arr_out

        vr, vc = np.where(mask)

        win_tr = ds.window_transform(window)
        col_coords = win_tr[2] + np.arange(window.width) * win_tr[0]
        row_coords = win_tr[5] + np.arange(window.height) * win_tr[4]

        row_f = np.clip(
            ((bounds_canvas_coords[3] - row_coords) / (bounds_canvas_coords[3] - bounds_canvas_coords[1])) * num_row
            - 0.5,
            0,
            num_row - 1,
        )
        col_f = np.clip(
            ((col_coords - bounds_canvas_coords[0]) / (bounds_canvas_coords[2] - bounds_canvas_coords[0])) * num_col
            - 0.5,
            0,
            num_col - 1,
        )

        ref = _weighted_bilinear_interpolation(
            _worker_dataset_cache["block_ref_mean"][:, :, band_idx], col_f[vc], row_f[vr]
        )
        loc = _weighted_bilinear_interpolation(
            _worker_dataset_cache["block_loc_mean"][:, :, band_idx], col_f[vc], row_f[vr]
        )

        if correction_method == "gamma":
            smallest = min(np.min(arr_in[mask]), np.min(ref), np.min(loc))
            if smallest <= 0:
                offset = abs(smallest) + 1
                arr_out[mask], gammas = _apply_gamma_correction(
                    arr_in[mask] + offset,
                    ref + offset,
                    loc + offset,
                    alpha,
                )
                arr_out[mask] -= offset
            else:
                arr_out[mask], gammas = _apply_gamma_correction(arr_in[mask], ref, loc, alpha)
        elif correction_method == "linear":
            gammas = ref / loc
            arr_out[mask] = arr_in[mask] * gammas
        else: raise ValueError('Invalid correction method')

        # if save_block_maps:
        #     gammas_array = np.full((*arr_in.shape, num_bands), np.nan, dtype=calculation_dtype)
        #     gammas_array[..., band_idx][mask] = gammas
        #     _download_block_map(
        #         gammas_array,
        #         bounding_rect_image,
        #         os.path.join(output_image_folder, "Gamma", out_name + f"_Gamma.tif"),
        #         projection,
        #         calculation_dtype,
        #         nodata_val,
        #         arr_in.shape[1],
        #         arr_in.shape[0],
        #         (band_idx,),
        #         window,
        #     )

        return window, band_idx, arr_out
    except Exception as e:
        print(f"\nWorker failed on band {band_idx}, window {window}: {e}")
        traceback.print_exc()
        raise


def _get_bounding_rectangle(
    image_paths: List[str]
    ) -> Tuple[float, float, float, float]:
    """
    Calculates the bounding rectangle that encompasses all input raster images.

    Args:
        image_paths (List[str]): List of raster file paths.

    Returns:
        Tuple[float, float, float, float]: (min_x, min_y, max_x, max_y) of the combined extent.
    """

    x_mins, y_mins, x_maxs, y_maxs = [], [], [], []

    for path in image_paths:
        with rasterio.open(path) as src:
            bounds = src.bounds
            x_mins.append(bounds.left)
            y_mins.append(bounds.bottom)
            x_maxs.append(bounds.right)
            y_maxs.append(bounds.top)

    return (min(x_mins), min(y_mins), max(x_maxs), max(y_maxs))


def _compute_mosaic_coefficient_of_variation(
    image_paths: List[str],
    nodata_value: float,
    reference_std: float = 45.0,
    reference_mean: float = 125.0,
    base_block_size: Tuple[int, int] = (10, 10),
    band_index: int = 1,
    calculation_dtype="float32",
    ) -> Tuple[int, int]:
    """
    Estimates block size for local adjustment using the coefficient of variation across input images.

    Args:
        image_paths (List[str]): List of input raster file paths.
        nodata_value (float): Value representing NoData in the input rasters.
        reference_std (float, optional): Reference standard deviation for comparison. Defaults to 45.0.
        reference_mean (float, optional): Reference mean for comparison. Defaults to 125.0.
        base_block_size (Tuple[int, int], optional): Base block size (rows, cols). Defaults to (10, 10).
        band_index (int, optional): Band index to use for statistics (1-based). Defaults to 1.
        calculation_dtype (str, optional): Data type for computation. Defaults to "float32".

    Returns:
        Tuple[int, int]: Estimated block size (rows, cols) adjusted based on coefficient of variation.
    """
    all_pixels = []

    for path in image_paths:
        try:
            with rasterio.open(path) as src:
                arr = src.read(band_index).astype(calculation_dtype)
                if nodata_value is not None:
                    arr = arr[arr != nodata_value]
                if arr.size > 0:
                    all_pixels.append(arr)
        except Exception:
            continue

    if not all_pixels:
        return base_block_size

    combined = np.concatenate(all_pixels)
    mean_val = np.mean(combined)
    std_val = np.std(combined)

    if mean_val == 0:
        return base_block_size

    catar = std_val / mean_val
    print(f"Mosaic coefficient of variation (CAtar) = {catar:.4f}")

    caref = reference_std / reference_mean
    r = catar / caref if caref != 0 else 1.0

    m, n = base_block_size
    return max(1, int(round(r * m))), max(1, int(round(r * n)))


def _compute_local_blocks(
    input_image_pairs: dict[str, str],
    bounds_canvas_coords: Tuple[float, float, float, float],
    num_row: int,
    num_col: int,
    num_bands: int,
    window_size: Optional[Tuple[int, int] | Literal["block"]] | None,
    debug_logs: bool,
    nodata_value: float,
    calculation_dtype: str,
    vector_mask_path: Tuple[Literal["include", "exclude"], str] | Tuple[Literal["include", "exclude"], str, str] | None,
    block_valid_pixel_threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Computes local block-wise mean values and valid pixel counts for each input image.

    Args:
        input_image_pairs (dict[str, str]): Dictionary mapping image names to file paths.
        bounds_canvas_coords (tuple): (minx, miny, maxx, maxy) of the full mosaic extent.
        num_row (int): Number of block rows in the canvas.
        num_col (int): Number of block columns in the canvas.
        num_bands (int): Number of bands per image.
        window_size (tuple[int, int] | Literal["block"] | None): Tiling mode for reading.
        debug_logs (bool): Whether to print debug statements.
        nodata_value (float): Value representing NoData in input rasters.
        calculation_dtype (str): Numpy dtype string used for internal calculations.
        vector_mask_path (tuple): mode, path, optional_field (example: "include", "/path/to/mask.gpkg", "image_field").
        block_valid_pixel_threshold (float): Minimum fraction of valid pixels required per block (0–1).

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
            - Block means per image (shape: num_row × num_col × num_bands)
            - Valid pixel counts per image (same shape)
    """

    block_value_sum = {}
    block_pixel_count = {}

    x_min, y_min, x_max, y_max = bounds_canvas_coords
    block_width = (x_max - x_min) / num_col
    block_height = (y_max - y_min) / num_row

    for name, image_path in input_image_pairs.items():
        if debug_logs: print(f'    {name}')
        output_shape = (num_row, num_col, num_bands)
        block_value_sum[name] = np.zeros(output_shape, dtype=calculation_dtype)
        block_pixel_count[name] = np.zeros(output_shape, dtype=calculation_dtype)

        with rasterio.open(image_path) as dataset:
            pixel_size_x, pixel_size_y = abs(dataset.transform.a), abs(dataset.transform.e)
            min_required_pixels = block_valid_pixel_threshold * (block_width * block_height) / (pixel_size_x * pixel_size_y)

            # Load vector mask if applicable
            geoms = None
            invert = None
            if vector_mask_path:
                mode, path, *field = vector_mask_path
                invert = mode == "exclude"
                field_name = field[0] if field else None

                with fiona.open(path, "r") as vector:
                    if field_name:
                        geoms = [
                            feat["geometry"]
                            for feat in vector
                            if field_name in feat["properties"] and name in str(feat["properties"][field_name])
                        ]
                    else:
                        geoms = [feat["geometry"] for feat in vector]
            if geoms and debug_logs: print("        Applied mask")

            windows_to_process = [
                (None, None, win) for win in _resolve_windows(
                    dataset,
                    window_size,
                    block_params=(num_row, num_col, dataset.bounds) if window_size == "block" else None
                )
            ]

            for band_index in range(num_bands):
                for window_index, (_, _, window) in enumerate(windows_to_process):
                    result = _process_local_block_window(
                        dataset_path=dataset.name,
                        band_index=band_index,
                        window=window,
                        geoms=geoms,
                        invert=invert,
                        nodata_value=nodata_value,
                        calculation_dtype=calculation_dtype,
                        transform=dataset.transform,
                        block_shape=(num_row, num_col),
                        bounds_canvas_coords=(x_min, y_min, x_max, y_max),
                    )

                    if result is None:
                        continue

                    value_sum, value_count = result
                    block_value_sum[name][:, :, band_index] += value_sum
                    block_pixel_count[name][:, :, band_index] += value_count

    block_mean_result = {}
    for name in input_image_pairs:
        with np.errstate(invalid='ignore', divide='ignore'):
            block_mean_result[name] = np.where(
                block_pixel_count[name] >= min_required_pixels,
                block_value_sum[name] / block_pixel_count[name],
                np.nan,
            )

    return block_mean_result, block_pixel_count


def _process_local_block_window(
    dataset_path: str,
    band_index: int,
    window: Window,
    geoms: Optional[list],
    invert: bool,
    nodata_value: float,
    calculation_dtype: str,
    transform,
    block_shape: Tuple[int, int],
    bounds_canvas_coords: Tuple[float, float, float, float],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Processes a tile window and returns block-wise sums and counts.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of shape (num_row, num_col) with sums and counts.
    """
    num_row, num_col = block_shape
    x_min, y_min, x_max, y_max = bounds_canvas_coords
    block_width = (x_max - x_min) / num_col
    block_height = (y_max - y_min) / num_row

    with rasterio.open(dataset_path) as dataset:
        tile_data = dataset.read(band_index + 1, window=window).astype(calculation_dtype)

        if geoms:
            tile_transform = dataset.window_transform(window)
            mask = geometry_mask(
                geoms, transform=tile_transform, invert=not invert, out_shape=(int(window.height), int(window.width))
            )
            tile_data[~mask] = nodata_value

        valid_mask = tile_data != nodata_value
        valid_rows, valid_cols = np.where(valid_mask)
        if valid_rows.size == 0:
            return None

        pixel_x = window.col_off + valid_cols + 0.5
        pixel_y = window.row_off + valid_rows + 0.5
        coords_x, coords_y = dataset.transform * (pixel_x, pixel_y)
        coords_x = np.array(coords_x)
        coords_y = np.array(coords_y)

        col_blocks = np.clip(((coords_x - x_min) / block_width).astype(int), 0, num_col - 1)
        row_blocks = np.clip(((y_max - coords_y) / block_height).astype(int), 0, num_row - 1)

        pixel_values = tile_data[valid_rows, valid_cols]

        value_sum = np.zeros((num_row, num_col), dtype=calculation_dtype)
        value_count = np.zeros((num_row, num_col), dtype=calculation_dtype)

        np.add.at(value_sum, (row_blocks, col_blocks), pixel_values)
        np.add.at(value_count, (row_blocks, col_blocks), 1)

        return value_sum, value_count


def _weighted_bilinear_interpolation(
    C_B: np.ndarray,
    x_frac: np.ndarray,
    y_frac: np.ndarray,
    ):
    """
    Performs bilinear interpolation on a 2D array while handling NaN values using a validity mask.

    Args:
        C_B (np.ndarray): 2D array with possible NaNs to interpolate.
        x_frac (np.ndarray): Fractional x-coordinates for interpolation.
        y_frac (np.ndarray): Fractional y-coordinates for interpolation.

    Returns:
        np.ndarray: Interpolated values at the specified fractional coordinates, with NaNs preserved where data is invalid.
    """

    # Create a mask that is 1 where valid (not NaN), 0 where NaN
    mask = ~np.isnan(C_B)

    # Replace NaNs with 0 in the original data (just for the interpolation step)
    C_B_filled = np.where(mask, C_B, 0)

    # Interpolate the "filled" data
    interpolated_data = map_coordinates(
        C_B_filled, [y_frac, x_frac], order=1, mode="reflect"  # bilinear interpolation
    )

    # Interpolate the mask in the same way
    interpolated_mask = map_coordinates(
        mask.astype(float), [y_frac, x_frac], order=1, mode="reflect"
    )

    # Divide data by mask to get final result
    with np.errstate(divide="ignore", invalid="ignore"):
        interpolated_values = interpolated_data / interpolated_mask
        # Where the interpolated mask is 0, set output to NaN (no valid neighbors)
        interpolated_values[interpolated_mask == 0] = np.nan

    return interpolated_values


def _download_block_map(
    block_map: np.ndarray,
    bounding_rect: Tuple[float, float, float, float],
    output_image_path: str,
    projection: rasterio.CRS,
    dtype: str,
    nodata_value: float,
    width: int,
    height: int,
    write_bands: Tuple[int, ...] | None = None,
    window: Window | None = None,
    ):

    """
    Writes specified bands of a 3D block map to an existing or new raster, using the provided window.

    Args:
        block_map (np.ndarray): Array of shape (window.height, window.width, num_bands).
        bounding_rect (tuple): Full bounding box (minx, miny, maxx, maxy).
        output_image_path (str): Destination file path.
        projection (CRS): Output CRS.
        dtype (str): Data type to use for output.
        nodata_value (float): Nodata value to assign.
        window (Window): Rasterio window for partial writing.
        width (int): Width of full output image.
        height (int): Height of full output image.
        write_bands (Tuple[int, ...]): Band indices (1-based) to write to or None to write all.
    """

    num_bands = block_map.shape[2]
    if block_map.ndim != 3: raise ValueError("block_map must be a 3D array with shape (num_row, num_col, num_bands).")
    if not os.path.exists(os.path.dirname(output_image_path)): os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    if write_bands is None: write_bands = tuple(range(0, num_bands))
    if window is None: window = Window(0, 0, width, height)

    transform = from_origin(
        bounding_rect[0],
        bounding_rect[3],
        (bounding_rect[2] - bounding_rect[0]) / width,
        (bounding_rect[3] - bounding_rect[1]) / height,
    )

    with file_lock:
        if not os.path.exists(output_image_path):
            profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": num_bands,
                "dtype": dtype,
                "crs": projection,
                "transform": transform,
                "nodata": nodata_value,
            }
            with rasterio.open(output_image_path, "w", **profile) as dst:
                for b in range(1, num_bands + 1):
                    dst.write(np.full((height, width), nodata_value, dtype=dtype), b)

    with rasterio.open(output_image_path, "r+") as dst:
        for band_index in write_bands:
            dst.write(block_map[:, :, band_index].astype(dtype), band_index+1, window=window)


def _compute_block_size(
    input_image_array_path: list,
    target_blocks_per_image: int | float,
    bounds_canvas_coords: tuple,
    ):
    """
    Calculates the number of rows and columns for dividing a bounding rectangle into target-sized blocks.

    Args:
        input_image_array_path (list): List of image paths to determine total image count.
        target_blocks_per_image (int | float): Desired number of blocks per image.
        bounds_canvas_coords (tuple): Bounding box covering all images (minx, miny, maxx, maxy).

    Returns:
        Tuple[int, int]: Number of rows (num_row) and columns (num_col) for the block grid.
    """

    num_images = len(input_image_array_path)

    # Total target blocks scaled by the number of images
    total_blocks = target_blocks_per_image * num_images

    x_min, y_min, x_max, y_max = bounds_canvas_coords
    bounding_width = x_max - x_min
    bounding_height = y_max - y_min

    # Aspect ratio of the bounding rectangle
    aspect_ratio = bounding_width / bounding_height

    # Start by assuming the number of columns (num_col)
    # We'll calculate num_col as the square root of total blocks scaled to the aspect ratio
    num_col = math.sqrt(total_blocks * aspect_ratio)
    num_col = max(1, round(num_col))  # Ensure at least one column

    # Calculate the number of rows (num_row) to match the number of blocks
    num_row = max(1, round(total_blocks / num_col))

    # Adjust for the closest fit to ensure num_row * num_col ≈ total_blocks
    while num_row * num_col < total_blocks:
        if bounding_width > bounding_height:
            num_col += 1
        else:
            num_row += 1

    while num_row * num_col > total_blocks:
        if bounding_width > bounding_height:
            num_col -= 1
        else:
            num_row -= 1

    return num_row, num_col


def _apply_gamma_correction(
    arr_in: np.ndarray,
    Mrefs: np.ndarray,
    Mins: np.ndarray,
    alpha: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies gamma correction to input pixel values based on reference and input block means.

    Args:
        arr_in (np.ndarray): Input pixel values to be corrected.
        Mrefs (np.ndarray): Reference block means.
        Mins (np.ndarray): Local block means of the input image.
        alpha (float, optional): Scaling factor applied to the corrected output. Defaults to 1.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Gamma-corrected pixel values.
            - Gamma values used in the correction.

    Raises:
        ValueError: If any value in Mins is zero or negative.
    """

    # Ensure no division by zero
    if np.any(Mins <= 0):
        raise ValueError(
            "Mins_valid contains zero or negative values, which are invalid for logarithmic operations. Maybe offset isnt working correctly."
        )

    # Compute gamma values (log(M_ref) / log(M_in))
    gammas = np.log(Mrefs) / np.log(Mins)

    # Apply gamma correction: P_res = alpha * P_in^gamma
    arr_out_valid = alpha * (arr_in**gammas)

    return arr_out_valid, gammas


def _smooth_array(
    input_array: np.ndarray,
    nodata_value: Optional[float] = None,
    scale_factor: float = 1.0,
    ) -> np.ndarray:
    """
    Applies Gaussian smoothing to an array while preserving NoData regions.

    Args:
        input_array (np.ndarray): 2D array to be smoothed.
        nodata_value (Optional[float], optional): Value representing NoData. Treated as NaN during smoothing. Defaults to None.
        scale_factor (float, optional): Sigma value for the Gaussian filter. Controls smoothing extent. Defaults to 1.0.

    Returns:
        np.ndarray: Smoothed array with NoData regions preserved or restored.
    """

    # Replace nodata_value with NaN for consistency
    if nodata_value is not None:
        input_array = np.where(input_array == nodata_value, np.nan, input_array)

    # Create a mask for valid (non-NaN) values
    valid_mask = ~np.isnan(input_array)

    # Replace NaN values with 0 to avoid affecting the smoothing
    array_with_nan_replaced = np.nan_to_num(input_array, nan=0.0)

    # Apply Gaussian smoothing
    smoothed = gaussian_filter(array_with_nan_replaced, sigma=scale_factor)

    # Normalize by the valid mask smoothed with the same kernel
    normalization_mask = gaussian_filter(valid_mask.astype(float), sigma=scale_factor)

    # Avoid division by zero in areas where the valid mask is 0
    smoothed_normalized = np.where(
        normalization_mask > 0, smoothed / normalization_mask, np.nan
    )

    # Reapply the nodata value (if specified) for output consistency
    if nodata_value is not None:
        smoothed_normalized = np.where(valid_mask, smoothed_normalized, nodata_value)

    return smoothed_normalized


def _init_worker(
    img_path: str,
    ref_shm_name: str,
    loc_shm_name: str,
    ref_shape: tuple,
    loc_shape: tuple,
    dtype_name: str
    ):

    global _worker_dataset_cache
    _worker_dataset_cache["ds"] = rasterio.open(img_path, "r")

    dtype = np.dtype(dtype_name)

    # Store SharedMemory objects to prevent premature GC
    ref_shm = shared_memory.SharedMemory(name=ref_shm_name)
    loc_shm = shared_memory.SharedMemory(name=loc_shm_name)

    _worker_dataset_cache["ref_shm"] = ref_shm
    _worker_dataset_cache["loc_shm"] = loc_shm

    _worker_dataset_cache["block_ref_mean"] = np.ndarray(ref_shape, dtype=dtype, buffer=ref_shm.buf)
    _worker_dataset_cache["block_loc_mean"] = np.ndarray(loc_shape, dtype=dtype, buffer=loc_shm.buf)