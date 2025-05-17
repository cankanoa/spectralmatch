import multiprocessing as mp
import math
import os
import numpy as np
import rasterio
import traceback
import gc

from osgeo import gdal
from scipy.ndimage import map_coordinates, gaussian_filter
from rasterio.windows import from_bounds, Window
from rasterio.transform import from_origin
from rasterio.crs import CRS
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Optional, List, Literal, Union
from multiprocessing import Lock
from multiprocessing import shared_memory

from ..utils import _check_raster_requirements, _get_nodata_value, _create_windows, _choose_context
from ..handlers import _resolve_input_output_paths

# Multiprocessing setup
_worker_dataset_cache = {}
file_lock = Lock()

def local_block_adjustment(
    input_images: str | List[str],
    output_images: Tuple[str, str] | List[str],
    *,
    custom_nodata_value: float | None = None,
    number_of_blocks: int | Tuple[int, int] | Literal["coefficient_of_variation"] = 100,
    alpha: float = 1.0,
    calculation_dtype_precision: str = "float32",
    output_dtype: str = "int16",
    debug_logs: bool = False,
    window_size: int | Tuple[int, int] | None | Literal["block"] = None,
    correction_method: Literal["gamma", "linear"] = "gamma",
    parallel_workers: Literal["cpu"] | int | None = None,
    save_block_maps: Tuple[str, str] | None = None,
    load_block_maps: Tuple[str, List[str]] | Tuple[str, None]| Tuple[None, List[str]] | None = None,
    override_bounds_canvas_coords: Tuple[float, float, float, float] | None = None,
    )-> list:
    """
    Performs local radiometric adjustment on a set of raster images using block-based statistics.

    Args:
        input_images (str | List[str]): A folder path containing `.tif` files to search for or a list of input image paths.
        output_images (Tuple[str, str] | List[str]): Either a tuple of (output_folder, suffix) to generate output paths from, or a list of output image paths. If a list is provided, its length must match the number of input images.
        custom_nodata_value (float | None, optional): Overrides detected NoData value. Defaults to None.
        number_of_blocks (int | tuple | Literal["coefficient_of_variation"]): int as a target of blocks per image, tuple to set manually set total blocks width and height, coefficient_of_variation to find the number of blocks based on this metric.
        alpha (float, optional): Blending factor between reference and local means. Defaults to 1.0.
        calculation_dtype_precision (str, optional): Precision for internal calculations. Defaults to "float32".
        output_dtype (str, optional): Data type for output rasters. Defaults to "float32".
        debug_logs (bool, optional): If True, prints progress. Defaults to False.
        window_size (int | Tuple[int, int] | None | Literal["block"]): Tile size for processing: int for square tiles, (width, height) for custom size, None for full image, or "block" to set as the size of the block map. Defaults to None.
        correction_method (Literal["gamma", "linear"], optional): Local correction method. Defaults to "gamma".
        parallel_workers (Literal["cpu"] | int | None): If set, enables multiprocessing. "cpu" = all cores, int = specific count, None = no parallel processing. Defaults to None.
        save_block_maps (tuple(str, str) | None): If enabled, saves block maps for review, to resume processing later, or to add additional images to the reference map. First str is a path to save the global block map. The second is a path to save the local block maps with the name of each image appended to its basename (because there are multiple local maps).
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

    Returns:
        List[str]: Paths to the locally adjusted output raster images.
    """

    print("Start local block adjustment")

    input_image_pairs, output_image_pairs = _resolve_input_output_paths(input_images, output_images)
    input_image_names = list(input_image_pairs.keys())
    input_image_paths = list(input_image_pairs.values())

    _check_raster_requirements(input_image_paths, debug_logs)

    if isinstance(window_size, int): window_size = (window_size, window_size)
    nodata_val = _get_nodata_value(input_image_paths, custom_nodata_value)
    projection = rasterio.open(input_image_paths[0]).crs
    if debug_logs: print(f"Global nodata value: {nodata_val}")
    with rasterio.open(input_image_paths[0]) as ds:num_bands = ds.count

    # Load data from precomputed block maps if set
    if load_block_maps:
        loaded_block_local_means, loaded_block_reference_mean, loaded_num_row, loaded_num_col, loaded_bounds_canvas_coords = _get_pre_computed_block_maps(load_block_maps, calculation_dtype_precision, debug_logs)
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
        print(f"Total images: input images: {len(input_image_names)}, loaded local block maps: {len(loaded_names) if load_block_maps else 0}")
        print(f"    Matched local block maps (to override) ({len(matched)}):", sorted(matched))
        print(f"    Only in loaded local block maps (to use) ({len(only_loaded)}):", sorted(only_loaded))
        print(f"    Only in input (to compute) ({len(only_input)}):", sorted(only_input))

    # Unpack path to save block maps
    if save_block_maps:
        if not isinstance(save_block_maps, tuple):
            raise TypeError("Expected a tuple to save block maps like this: (reference_map_path, local_map_path), where reference_map_path is a normal path and local_map_path will have the names of images prepended to it.")
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

    if debug_logs: print("Computing local block maps")

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
            calculation_dtype_precision,
        )
        overlap = set(block_local_means) & set(local_blocks_to_load)
        if overlap:
            raise ValueError(f"Duplicate keys when merging loaded and computed blocks: {overlap}")

        block_local_means = {**block_local_means, **local_blocks_to_load}
    else:
        block_local_means = local_blocks_to_load


    bounds_images_block_space = get_bounding_rect_images_block_space(block_local_means)

    # Compute reference block
    if debug_logs: print("Computing reference block map")
    if block_reference_mean is None:
        block_reference_mean = _compute_reference_blocks(
            block_local_means,
            calculation_dtype_precision,
            )

    if save_block_maps:
        _download_block_map(
            np.nan_to_num(block_reference_mean, nan=nodata_val),
            bounds_canvas_coords,
            reference_map_path,
            projection,
            calculation_dtype_precision,
            nodata_val,
            num_col,
            num_row,
        )
        for name, block_local_mean in block_local_means.items():
            _download_block_map(
                np.nan_to_num(block_local_mean, nan=nodata_val),
                bounds_canvas_coords,
                os.path.join(os.path.dirname(local_map_path), name + os.path.splitext(os.path.basename(local_map_path))[0] + ".tif"),
                projection,
                calculation_dtype_precision,
                nodata_val,
                num_col,
                num_row,
            )
            # _download_block_map(
            #     np.nan_to_num(block_local_count, nan=nodata_val),
            #     bounds_canvas_coords,
            #     os.path.join(output_image_folder, "BlockLocalCount", f"{input_image_name}_BlockLocalCount.tif"),
            #     projection,
            #     calculation_dtype_precision,
            #     nodata_val,
            #     num_col,
            #     num_row,
            # )

    # block_local_mean = _smooth_array(block_local_mean, nodata_value=global_nodata_value)

    if parallel_workers == "cpu":
        parallel = True
        max_workers = mp.cpu_count()
    elif isinstance(parallel_workers, int) and parallel_workers > 0:
        parallel = True
        max_workers = parallel_workers
    else:
        parallel = False
        max_workers = None

    if debug_logs: print(f"Computing local correction, applying, and saving")
    out_paths: List[str] = []
    for name, img_path in input_image_pairs.items():
        in_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = output_image_pairs[name]
        out_name = os.path.splitext(os.path.basename(out_path))[0]
        out_paths.append(str(out_path))

        if debug_logs: print(f"    Processing {in_name}")
        with rasterio.open(img_path) as src:
            meta = src.meta.copy()
            meta.update({"count": num_bands, "dtype": output_dtype, "nodata": nodata_val})
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
                                        calculation_dtype_precision,
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
                                calculation_dtype_precision,
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
    print("Finished local block adjustment")
    return out_paths


def _get_pre_computed_block_maps(
    load_block_maps: Tuple[Optional[str], Optional[List[str]]],
    calculation_dtype_precision: str,
    debug_logs: bool,
) -> Tuple[dict[str, np.ndarray], Optional[np.ndarray], Optional[int], Optional[int], Optional[Tuple[float, float, float, float]]]:
    """
    Load pre-computed block mean maps from files.

    Args:
        load_block_maps (Tuple[str, List[str]] | Tuple[str, None] | Tuple[None, List[str]]):
            - Tuple[str, List[str]]: Load both reference and local block maps.
            - Tuple[str, None]: Load only the reference block map.
            - Tuple[None, List[str]]: Load only the local block maps.
        calculation_dtype_precision (str): Numpy dtype to use for reading.
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
            ref_data = src.read().astype(calculation_dtype_precision).transpose(1, 2, 0)
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
                data = src.read().astype(calculation_dtype_precision).transpose(1, 2, 0)
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
    calculation_dtype_precision: str,
) -> np.ndarray:
    """
    Computes reference block means across images by averaging non-NaN local block means.

    Args:
        block_local_means (dict[str, np.ndarray]): Per-image block mean arrays.
        calculation_dtype_precision (str): Numpy dtype for output array.

    Returns:
        np.ndarray: Reference block map of shape (num_row, num_col, num_bands)
    """
    shape = next(iter(block_local_means.values())).shape
    stacked = np.stack(list(block_local_means.values()), axis=0)  # shape: (num_images, H, W, B)
    with np.errstate(invalid='ignore'):
        ref_block_mean = np.nanmean(stacked, axis=0).astype(calculation_dtype_precision)
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
    calculation_dtype_precision: str,
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
        calculation_dtype_precision (str): Internal computation precision (e.g., "float32").

    Returns:
        tuple: (Window, band index, corrected tile as np.ndarray)
    """
    try:
        # if debug_logs: print(f"b{band_idx}w{w_id}[{window.col_off}:{window.row_off} {window.width}x{window.height}], ", end="", flush=True)

        ds = _worker_dataset_cache["ds"]
        arr_in = ds.read(band_idx + 1, window=window).astype(calculation_dtype_precision)
        arr_out = np.full_like(arr_in, nodata_val, dtype=calculation_dtype_precision)

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
        #     gammas_array = np.full((*arr_in.shape, num_bands), np.nan, dtype=calculation_dtype_precision)
        #     gammas_array[..., band_idx][mask] = gammas
        #     _download_block_map(
        #         gammas_array,
        #         bounding_rect_image,
        #         os.path.join(output_image_folder, "Gamma", out_name + f"_Gamma.tif"),
        #         projection,
        #         calculation_dtype_precision,
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
    calculation_dtype_precision="float32",
    ) -> Tuple[int, int]:
    """
    Estimates block size for local adjustment using the coefficient of variation across input images.

    Args:
        image_paths (List[str]): List of input raster file paths.
        nodata_value (float): Value representing NoData in the input rasters.
        reference_std (float, optional): Reference standard deviation for comparison. Defaults to 45.0.
        reference_mean (float, optional): Reference mean for comparison. Defaults to 125.0.
        base_block_size (Tuple[int, int], optional): Base block size (rows, cols). Defaults to (10, 10).
        band_index (int, optional): Band index to use for statistics. Defaults to 1.
        calculation_dtype_precision (str, optional): Data type for computation. Defaults to "float32".

    Returns:
        Tuple[int, int]: Estimated block size (rows, cols) adjusted based on coefficient of variation.
    """

    all_pixels = []

    # Collect pixel values from all images
    for path in image_paths:
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            continue
        band = ds.GetRasterBand(band_index)
        arr = band.ReadAsArray().astype(calculation_dtype_precision)
        if nodata_value is not None:
            mask = arr != nodata_value
            arr = arr[mask]
        all_pixels.append(arr)
        ds = None

    # If no valid pixels, return defaults
    if len(all_pixels) == 0:
        return 0.0, base_block_size[0], base_block_size[1]

    # Combine pixel values and compute statistics
    combined = np.concatenate(all_pixels)
    mean_val = np.mean(combined)
    std_val = np.std(combined)
    if mean_val == 0:
        return 0.0, base_block_size[0], base_block_size[1]

    catar = std_val / mean_val
    print(f"Mosaic coefficient of variation (CAtar) = {catar:.4f}")

    # Compute reference coefficient and adjustment ratio
    caref = reference_std / reference_mean
    r = catar / caref if caref != 0 else 1.0

    # Adjust block size
    m, n = base_block_size
    num_row = max(1, int(round(r * m)))
    num_col = max(1, int(round(r * n)))

    return num_row, num_col

def _compute_local_blocks(
    input_image_pairs: dict[str, str],
    bounds_canvas_coords: Tuple[float, float, float, float],
    num_row: int,
    num_col: int,
    num_bands: int,
    window_size: Optional[Tuple[int, int] | Literal["block"]] | None,
    debug_logs: bool,
    nodata_value: float,
    calculation_dtype_precision: str,
    valid_pixel_threshold: float = 0.001,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Computes local block-wise mean values and valid pixel counts for each input image.

    Args:
        input_image_pairs (dict[str, str]): Dictionary mapping image names to file paths.
        bounds_canvas_coords (tuple): (minx, miny, maxx, maxy) of the full mosaic extent.
        num_row (int): Number of block rows in the canvas.
        num_col (int): Number of block columns in the canvas.
        num_bands (int): Number of bands per image.
        window_size (tuple[int, int] | Literal["block"] | None): Tiling mode for reading:
            - None: full image,
            - "block": uses block-aligned windows based on canvas grid,
            - (width, height): custom tile size in pixels.
        debug_logs (bool): Whether to print debug statements.
        nodata_value (float): Value representing NoData in input rasters.
        calculation_dtype_precision (str): Numpy dtype string used for internal calculations.
        valid_pixel_threshold (float): Minimum fraction of valid pixels required per block (0–1).

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
    min_required_pixels = valid_pixel_threshold * block_width * block_height

    for name, image_path in input_image_pairs.items():
        if debug_logs: print(f'    Processing: {name}')
        output_shape = (num_row, num_col, num_bands)
        block_value_sum[name] = np.zeros(output_shape, dtype=calculation_dtype_precision)
        block_pixel_count[name] = np.zeros(output_shape, dtype=calculation_dtype_precision)

        with rasterio.open(image_path) as dataset:
            dataset_bounds = dataset.bounds
            tiles_to_process: List[Tuple[Optional[int], Optional[int], Window]] = []

            if window_size is None:
                full_window = Window(0, 0, dataset.width, dataset.height)
                tiles_to_process.append((None, None, full_window))
            elif window_size == "block":
                for row_idx in range(num_row):
                    for col_idx in range(num_col):
                        block_x0 = x_min + col_idx * block_width
                        block_x1 = block_x0 + block_width
                        block_y1 = y_max - row_idx * block_height
                        block_y0 = block_y1 - block_height

                        if (block_x1 <= dataset_bounds.left or block_x0 >= dataset_bounds.right or
                            block_y1 <= dataset_bounds.bottom or block_y0 >= dataset_bounds.top):
                            continue

                        intersected_window = from_bounds(
                            max(block_x0, dataset_bounds.left),
                            max(block_y0, dataset_bounds.bottom),
                            min(block_x1, dataset_bounds.right),
                            min(block_y1, dataset_bounds.top),
                            transform=dataset.transform,
                        )
                        tiles_to_process.append((row_idx, col_idx, intersected_window))
            elif isinstance(window_size, tuple):
                tile_width, tile_height = window_size
                for tile in _create_windows(dataset.width, dataset.height, tile_width, tile_height):
                    tiles_to_process.append((None, None, tile))

            for band_index in range(num_bands):
                for tile_index, (row_idx, col_idx, tile_window) in enumerate(tiles_to_process):
                    tile_data = dataset.read(band_index + 1, window=tile_window).astype(calculation_dtype_precision)

                    if nodata_value is not None:
                        valid_mask = tile_data != nodata_value
                    else:
                        valid_mask = np.ones_like(tile_data, dtype=bool)

                    valid_rows, valid_cols = np.where(valid_mask)
                    if valid_rows.size == 0:
                        continue

                    pixel_x = tile_window.col_off + valid_cols + 0.5
                    pixel_y = tile_window.row_off + valid_rows + 0.5
                    coords_x, coords_y = dataset.transform * (pixel_x, pixel_y)
                    coords_x = np.array(coords_x)
                    coords_y = np.array(coords_y)

                    col_blocks = np.clip(((coords_x - x_min) / block_width).astype(int), 0, num_col - 1)
                    row_blocks = np.clip(((y_max - coords_y) / block_height).astype(int), 0, num_row - 1)

                    pixel_values = tile_data[valid_rows, valid_cols]
                    np.add.at(block_value_sum[name][:, :, band_index], (row_blocks, col_blocks), pixel_values)
                    np.add.at(block_pixel_count[name][:, :, band_index], (row_blocks, col_blocks), 1)

    block_mean_result = {}
    for name in input_image_pairs:
        with np.errstate(invalid='ignore', divide='ignore'):
            block_mean_result[name] = np.where(
                block_pixel_count[name] >= min_required_pixels,
                block_value_sum[name] / block_pixel_count[name],
                np.nan,
            )

    return block_mean_result, block_pixel_count



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