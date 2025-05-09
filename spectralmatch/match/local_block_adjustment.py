import multiprocessing as mp
import math
import gc
import os
import numpy as np
import rasterio
import traceback

from osgeo import gdal
from scipy.ndimage import map_coordinates, gaussian_filter
from rasterio.windows import Window
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.errors import RasterioIOError
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Optional, List, Literal

from ..utils import _check_raster_requirements, _get_nodata_value, _create_windows, _choose_context
_worker_dataset_cache = {}

def local_block_adjustment(
    input_image_paths: List[str],
    output_image_folder: str,
    *,
    output_local_basename: str = "_local",
    custom_nodata_value: float | None = None,
    target_blocks_per_image: int = 100,
    alpha: float = 1.0,
    calculation_dtype_precision: str = "float32",
    output_dtype: str = "float32",
    projection: str = "EPSG:4326",
    debug_mode: bool = False,
    tile_width_and_height_tuple: Optional[Tuple[int, int]] = None,
    correction_method: Literal["gamma", "linear"] = "gamma",
    parallel: bool = False,
    max_workers: int | None = None,
    )-> list:
    """
    Performs local radiometric adjustment on a set of raster images using block-based statistics.

    Args:
        input_image_paths (List[str]): List of raster image paths to adjust.
        output_image_folder (str): Folder to save corrected output rasters.
        output_local_basename (str, optional): Suffix for output filenames. Defaults to "_local".
        custom_nodata_value (float | None, optional): Overrides detected NoData value. Defaults to None.
        target_blocks_per_image (int, optional): Approximate number of blocks per image. Defaults to 100.
        alpha (float, optional): Blending factor between global and local means. Defaults to 1.0.
        calculation_dtype_precision (str, optional): Precision for internal calculations. Defaults to "float32".
        output_dtype (str, optional): Data type for output rasters. Defaults to "float32".
        projection (str, optional): CRS projection string for output block maps. Defaults to "EPSG:4326".
        debug_mode (bool, optional): If True, saves intermediate block maps and prints progress. Defaults to False.
        tile_width_and_height_tuple (Tuple[int, int], optional): Tile size for block-wise correction. Defaults to None.
        correction_method (Literal["gamma", "linear"], optional): Local correction method. Defaults to "gamma".
        parallel (bool, optional): If True, enables multiprocessing. Defaults to False.
        max_workers (int | None, optional): Max number of parallel workers. Defaults to number of CPUs.

    Returns:
        List[str]: Paths to the locally adjusted output raster images.
    """

    print("Start local block adjustment")
    _check_raster_requirements(input_image_paths, debug_mode)

    nodata_val = _get_nodata_value(input_image_paths, custom_nodata_value)
    if debug_mode: print(f"Global nodata value: {nodata_val}")

    out_img_dir = os.path.join(output_image_folder, "Images")
    if not os.path.exists(out_img_dir): os.makedirs(out_img_dir)

    bounding_rect = _get_bounding_rectangle(input_image_paths)
    M, N = _compute_block_size(input_image_paths, target_blocks_per_image, bounding_rect)
    # M, N = _compute_mosaic_coefficient_of_variation(input_image_paths, global_nodata_value) # Aproach from the paper to compute bock size

    with rasterio.open(input_image_paths[0]) as ds:
        num_bands = ds.count

    if debug_mode: print("Computing global reference block map")
    block_ref_mean, _ = _compute_blocks(
        input_image_paths,
        bounding_rect,
        M,
        N,
        num_bands,
        nodata_value=nodata_val,
        tile_width_and_height_tuple=tile_width_and_height_tuple,
    )

    if debug_mode:
        _download_block_map(
            block_map=np.nan_to_num(block_ref_mean, nan=nodata_val),
            bounding_rect=bounding_rect,
            output_image_path= os.path.join(output_image_folder, "BlockReferenceMean", "BlockReferenceMean.tif"),
            nodata_value=nodata_val,
            projection=projection,
        )

    if parallel and max_workers is None:
        max_workers = mp.cpu_count()

    out_paths: List[str] = []
    for img_path in input_image_paths:
        in_name = os.path.splitext(os.path.basename(img_path))[0]
        out_name = os.path.splitext(os.path.basename(img_path))[0] + output_local_basename
        out_path = os.path.join(out_img_dir, f"{out_name}.tif")
        out_paths.append(str(out_path))

        if debug_mode: print(f"Processing {in_name}")
        if debug_mode: print(f"Computing local block map")
        block_loc_mean, block_loc_count = _compute_blocks(
            [img_path],
            bounding_rect,
            M,
            N,
            num_bands,
            nodata_value=nodata_val,
            tile_width_and_height_tuple=tile_width_and_height_tuple,
        )

        # block_local_mean = _smooth_array(block_local_mean, nodata_value=global_nodata_value)

        if debug_mode:
            _download_block_map(
                block_map=np.nan_to_num(block_loc_mean, nan=nodata_val),
                bounding_rect=bounding_rect,
                output_image_path=os.path.join(output_image_folder, "BlockLocalMean", f"{out_name}_BlockLocalMean.tif"),
                nodata_value=nodata_val,
                projection=projection,
            )
            _download_block_map(
                block_map=np.nan_to_num(block_loc_count, nan=nodata_val),
                bounding_rect=bounding_rect,
                output_image_path=os.path.join(output_image_folder, "BlockLocalCount", f"{out_name}_BlockLocalCount.tif"),
                nodata_value=nodata_val,
                projection=projection,
            )

        if debug_mode: print(f"Computing local correction, applying, and saving")
        with rasterio.open(img_path) as src:
            meta = src.meta.copy()
            meta.update({"count": num_bands, "dtype": output_dtype, "nodata": nodata_val})
            with rasterio.open(out_path, "w", **meta) as dst:

                if tile_width_and_height_tuple:
                    tw, th = tile_width_and_height_tuple
                    windows = list(_create_windows(src.width, src.height, tw, th))
                else:
                    windows = [Window(0, 0, src.width, src.height)]
                if debug_mode: print(f"BandIDWindowID[xStart:yStart xSizeXySize] ({len(windows)} windows): ", end="")

                if parallel:
                    ctx = _choose_context(prefer_fork=True)

                    pool = ProcessPoolExecutor(
                        max_workers=max_workers,
                        mp_context=ctx,
                        initializer=_init_worker,
                        initargs=(img_path,),
                    )

                    futures = [
                        pool.submit(_compute_tile_local,
                                    w,
                                    b,
                                    M,
                                    N,
                                    bounding_rect,
                                    block_ref_mean,
                                    block_loc_mean,
                                    nodata_val,
                                    alpha,
                                    correction_method,
                                    calculation_dtype_precision,
                                    debug_mode,
                                    w_id,
                                    )
                        for b in range(num_bands)
                        for w_id, w in enumerate(windows)
                    ]
                    for fut in as_completed(futures):
                        win, b_idx, buf = fut.result()
                        dst.write(buf.astype(output_dtype), b_idx + 1, window=win)
                    pool.shutdown()
                else:
                    _init_worker(img_path)

                    for b in range(num_bands):
                        for w_id, win in enumerate(windows):
                            win_, b_idx, buf = _compute_tile_local(
                                win,
                                b,
                                M,
                                N,
                                bounding_rect,
                                block_ref_mean,
                                block_loc_mean,
                                nodata_val,
                                alpha,
                                correction_method,
                                calculation_dtype_precision,
                                debug_mode,
                                w_id,
                            )
                            dst.write(buf.astype(output_dtype), b_idx + 1, window=win_)
                if debug_mode: print()
    print("Finished local block adjustment")
    return out_paths


def _compute_tile_local(
    window: Window,
    band_idx: int,
    M: int,
    N: int,
    bounding_rect: tuple,
    block_ref_mean: np.ndarray,
    block_loc_mean: np.ndarray,
    nodata_val: float | int,
    alpha: float,
    correction_method: Literal["gamma", "linear"],
    calculation_dtype_precision: str,
    debug_mode: bool,
    w_id: int,
    ):
    """
    Applies local radiometric correction to a raster tile using bilinear interpolation of reference and local block means.

    Args:
        window (Window): Rasterio window defining the tile extent.
        band_idx (int): Index of the band to process.
        M (int): Number of rows in the block grid.
        N (int): Number of columns in the block grid.
        bounding_rect (tuple): Bounding rectangle of the full mosaic (minx, miny, maxx, maxy).
        block_ref_mean (np.ndarray): Global block mean reference array (shape: M x N x bands).
        block_loc_mean (np.ndarray): Local block mean array for the current image (shape: M x N x bands).
        nodata_val (float | int): Value representing NoData in the raster.
        alpha (float): Blending weight between global and local statistics.
        correction_method (Literal["gamma", "linear"]): Type of correction to apply.
        calculation_dtype_precision (str): Data type to use for internal computation (e.g., "float32").
        debug_mode (bool): If True, prints debug info.
        w_id (int): Which window is processing.

    Returns:
        tuple: (Window, band index, corrected tile as np.ndarray)
    """
    try:
        if debug_mode: print(f"b{band_idx}w{w_id}[{window.col_off}:{window.row_off} {window.width}x{window.height}], ", end="", flush=True)

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
            ((bounding_rect[3] - row_coords) / (bounding_rect[3] - bounding_rect[1])) * M
            - 0.5,
            0,
            M - 1,
        )
        col_f = np.clip(
            ((col_coords - bounding_rect[0]) / (bounding_rect[2] - bounding_rect[0])) * N
            - 0.5,
            0,
            N - 1,
        )

        ref = _weighted_bilinear_interpolation(
            block_ref_mean[:, :, band_idx], col_f[vc], row_f[vr]
        )
        loc = _weighted_bilinear_interpolation(
            block_loc_mean[:, :, band_idx], col_f[vc], row_f[vr]
        )
        # if debug_mode:
        #     gammas_array = np.full(arr_in.shape, global_nodata_value, dtype=calculation_dtype_precision)
        #     gammas_array[valid_rows, valid_cols] = gammas
        #     _download_block_map(
        #         block_map=gammas_array,
        #         bounding_rect=this_image_bounds,
        #         output_image_path=os.path.join(output_image_folder, "Gamma", out_name + f"_Gamma.tif"),
        #         projection=projection,
        #         nodata_value=global_nodata_value,
        #         output_bands_map=(b+1,),
        #         override_band_count=num_bands,
        #     )

        # if debug_mode:
        #     _download_block_map(
        #         block_map=np.where(valid_mask, 1, global_nodata_value),
        #         bounding_rect=this_image_bounds,
        #         output_image_path=os.path.join(output_image_folder, "ValidMasks", out_name + f"_ValidMask.tif"),
        #         projection=projection,
        #         nodata_value=global_nodata_value,
        #         output_bands_map=(b+1,),
        #         override_band_count=num_bands
        #     )

        # if debug_mode:
        #     _download_block_map(
        #         block_map=reference_band,
        #         bounding_rect=this_image_bounds,
        #         output_image_path=os.path.join(output_image_folder, "ReferenceBand", out_name + f"_ReferenceBand.tif"),
        #         projection=projection,
        #         nodata_value=global_nodata_value,
        #         output_bands_map=(b+1,),
        #         override_band_count=num_bands,
        #     )
        #
        # if debug_mode:
        #     _download_block_map(
        #         block_map=local_band,
        #         bounding_rect=this_image_bounds,
        #         output_image_path=os.path.join(output_image_folder, "LocalBand", out_name + f"_LocalBand.tif"),
        #         projection=projection,
        #         nodata_value=global_nodata_value,
        #         output_bands_map=(b+1,),
        #         override_band_count=num_bands,
        #     )

        if correction_method == "gamma":
            smallest = np.min([arr_in[mask], ref, loc])
            if smallest <= 0:
                offset = abs(smallest) + 1
                arr_out[mask], _ = _apply_gamma_correction(
                    arr_in[mask] + offset,
                    ref + offset,
                    loc + offset,
                    alpha,
                )
                arr_out[mask] -= offset
            else:
                arr_out[mask], _ = _apply_gamma_correction(arr_in[mask], ref, loc, alpha)
        elif correction_method == "linear":
            arr_out[mask] = arr_in[mask] * (ref / loc)
        else: raise ValueError('Invalid correction method')

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
    M = max(1, int(round(r * m)))
    N = max(1, int(round(r * n)))

    return M, N


def _compute_blocks(
    image_paths: List[str],
    bounding_rect: Tuple[float, float, float, float],
    M: int,
    N: int,
    num_bands: int,
    nodata_value: float = None,
    valid_pixel_threshold: float = 0.001,
    calculation_dtype_precision="float32",
    tile_width_and_height_tuple: tuple = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes block-wise mean values across multiple raster images for each band.

    Args:
        image_paths (List[str]): List of raster file paths.
        bounding_rect (Tuple[float, float, float, float]): Bounding box covering all rasters (minx, miny, maxx, maxy).
        M (int): Number of rows in the block grid.
        N (int): Number of columns in the block grid.
        num_bands (int): Number of bands to process.
        nodata_value (float, optional): Value representing NoData pixels. Defaults to None.
        valid_pixel_threshold (float, optional): Minimum valid pixel ratio to compute a block mean. Defaults to 0.001.
        calculation_dtype_precision (str, optional): Precision used during calculations. Defaults to "float32".
        tile_width_and_height_tuple (tuple, optional): Tile dimensions for memory-efficient processing. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - 3D array of block means with shape (M, N, bands), containing NaNs where insufficient data exists.
            - 3D array of valid pixel counts per block and band.
    """

    sum_of_means_3d = np.zeros((M, N, num_bands), dtype=calculation_dtype_precision)
    image_count_3d = np.zeros((M, N, num_bands), dtype=calculation_dtype_precision)
    sum_of_counts_3d = np.zeros((M, N, num_bands), dtype=calculation_dtype_precision)

    x_min, y_min, x_max, y_max = bounding_rect
    block_width = (x_max - x_min) / N
    block_height = (y_max - y_min) / M

    for path in image_paths:
        with rasterio.open(path) as data:
            transform = data.transform

            sum_map_3d_single = np.zeros((M, N, num_bands), dtype=calculation_dtype_precision)
            count_map_3d_single = np.zeros((M, N, num_bands), dtype=calculation_dtype_precision)

            for b in range(num_bands):
                sum_map_2d = np.zeros((M, N), dtype=calculation_dtype_precision)
                count_map_2d = np.zeros((M, N), dtype=calculation_dtype_precision)

                if tile_width_and_height_tuple:
                    windows = _create_windows(data.width, data.height, tile_width_and_height_tuple[0], tile_width_and_height_tuple[1])
                else:
                    windows = [Window(0, 0, data.width, data.height)]

                for win in windows:
                    arr = data.read(b + 1, window=win).astype(calculation_dtype_precision)
                    win_transform = data.window_transform(win)
                    height, width = arr.shape

                    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
                    x0, dx, _, y0, _, dy = win_transform.to_gdal()
                    xs = x0 + dx * (cols + 0.5)
                    ys = y0 + dy * (rows + 0.5)
                    xs = np.array(xs).flatten()
                    ys = np.array(ys).flatten()
                    arr_flat = arr.flatten()

                    if nodata_value is not None:
                        mask = arr_flat != nodata_value
                        xs, ys, arr_flat = xs[mask], ys[mask], arr_flat[mask]

                    block_cols = np.clip(((xs - x_min) / block_width).astype(int), 0, N - 1)
                    block_rows = np.clip(((y_max - ys) / block_height).astype(int), 0, M - 1)

                    np.add.at(sum_map_2d, (block_rows, block_cols), arr_flat)
                    np.add.at(count_map_2d, (block_rows, block_cols), 1)

                    del arr, arr_flat, xs, ys, mask, block_cols, block_rows
                    gc.collect()

                sum_map_3d_single[..., b] = sum_map_2d
                count_map_3d_single[..., b] = count_map_2d
                del sum_map_2d, count_map_2d
                gc.collect()

            block_map_3d_single = np.full((M, N, num_bands), np.nan, dtype=calculation_dtype_precision)
            for b in range(num_bands):
                count_band = count_map_3d_single[..., b]
                sum_band = sum_map_3d_single[..., b]
                valid = count_band > (valid_pixel_threshold * block_width * block_height)
                block_map_3d_single[valid, b] = sum_band[valid] / count_band[valid]

            valid_mask_3d = ~np.isnan(block_map_3d_single)
            sum_of_means_3d[valid_mask_3d] += block_map_3d_single[valid_mask_3d]
            image_count_3d[valid_mask_3d] += 1
            sum_of_counts_3d += count_map_3d_single

            del sum_map_3d_single, count_map_3d_single, block_map_3d_single, valid_mask_3d
            gc.collect()

    final_block_map = np.full((M, N, num_bands), np.nan, dtype=calculation_dtype_precision)
    positive_mask = image_count_3d > 0
    final_block_map[positive_mask] = sum_of_means_3d[positive_mask] / image_count_3d[positive_mask]

    return final_block_map, sum_of_counts_3d


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
    projection: str = "EPSG:4326",
    dtype="float32",
    nodata_value: float = None,
    output_bands_map: Optional[Tuple[int, ...]] = None,
    override_band_count: Optional[int] = None,
    ):
    """
    Saves a 2D or 3D block map as a raster file, writing to specific bands if requested.

    Args:
        block_map (np.ndarray): 2D or 3D array of block values to write.
        bounding_rect (Tuple[float, float, float, float]): Bounding box of the map (minx, miny, maxx, maxy).
        output_image_path (str): Path to save the output raster.
        projection (str, optional): CRS in EPSG format. Defaults to "EPSG:4326".
        dtype (str, optional): Data type for output raster. Defaults to "float32".
        nodata_value (float, optional): NoData value to use in the output. Defaults to None.
        output_bands_map (Optional[Tuple[int, ...]], optional): Band indices to write to. Defaults to sequential.
        override_band_count (Optional[int], optional): Total number of bands to reserve in the output. Defaults to inferred max.

    Returns:
        None

    Raises:
        ValueError: If band count does not match output_bands_map length.
        RuntimeError: If an existing file cannot be updated due to metadata mismatch.
    """

    output_dir = os.path.dirname(output_image_path)
    base = os.path.basename(output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x_min, y_min, x_max, y_max = bounding_rect
    M, N = block_map.shape[:2]
    num_bands = 1 if block_map.ndim == 2 else block_map.shape[2]

    if output_bands_map is not None and len(output_bands_map) != num_bands:
        raise ValueError("Length of output_bands_map must match the number of bands in block_map.")

    band_indices = output_bands_map if output_bands_map is not None else tuple(range(1, num_bands + 1))
    required_band_count = override_band_count if override_band_count is not None else max(band_indices)

    pixel_width = (x_max - x_min) / N
    pixel_height = (y_max - y_min) / M
    transform = from_origin(x_min, y_max, pixel_width, pixel_height)
    crs = CRS.from_string(projection)

    if os.path.exists(output_image_path):
        try:
            with rasterio.open(output_image_path, "r+") as dst:
                if (
                    dst.crs != crs or
                    dst.transform != transform or
                    dst.width != N or
                    dst.height != M or
                    dst.dtypes[0] != dtype
                ):
                    raise UserWarning(f"Metadata mismatch for existing file: {output_image_path}. Skipping.")

                if dst.count < max(band_indices):
                    raise RuntimeError(
                        f"Raster at {output_image_path} has {dst.count} bands but band {max(band_indices)} is requested. Cannot write."
                    )

                for idx, b in enumerate(band_indices):
                    band_data = block_map if num_bands == 1 else block_map[:, :, idx]
                    dst.write(band_data, b)
                return
        except RasterioIOError as e:
            raise RuntimeError(f"Error reading existing file {output_image_path}: {e}")

    # File does not exist — create new
    profile = {
        "driver": "GTiff",
        "height": M,
        "width": N,
        "count": required_band_count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata_value,
    }

    with rasterio.open(output_image_path, "w", **profile) as dst:
        # Fill all bands with nodata first if override_band_count was used
        if override_band_count is not None:
            for b in range(1, override_band_count + 1):
                dst.write(np.full((M, N), nodata_value, dtype=dtype), b)

        # Now write the actual bands to their mapped indices
        for idx, b in enumerate(band_indices):
            band_data = block_map if num_bands == 1 else block_map[:, :, idx]
            dst.write(band_data, b)

        print(f"Block map saved to: {output_image_path}")


def _compute_block_size(
    input_image_array_path: list,
    target_blocks_per_image: int | float,
    bounding_rect: tuple,
    ):
    """
    Calculates the number of rows and columns for dividing a bounding rectangle into target-sized blocks.

    Args:
        input_image_array_path (list): List of image paths to determine total image count.
        target_blocks_per_image (int | float): Desired number of blocks per image.
        bounding_rect (tuple): Bounding box covering all images (minx, miny, maxx, maxy).

    Returns:
        Tuple[int, int]: Number of rows (M) and columns (N) for the block grid.
    """

    num_images = len(input_image_array_path)

    # Total target blocks scaled by the number of images
    total_blocks = target_blocks_per_image * num_images

    x_min, y_min, x_max, y_max = bounding_rect
    bounding_width = x_max - x_min
    bounding_height = y_max - y_min

    # Aspect ratio of the bounding rectangle
    aspect_ratio = bounding_width / bounding_height

    # Start by assuming the number of columns (N)
    # We'll calculate N as the square root of total blocks scaled to the aspect ratio
    N = math.sqrt(total_blocks * aspect_ratio)
    N = max(1, round(N))  # Ensure at least one column

    # Calculate the number of rows (M) to match the number of blocks
    M = max(1, round(total_blocks / N))

    # Adjust for the closest fit to ensure M * N ≈ total_blocks
    while M * N < total_blocks:
        if bounding_width > bounding_height:
            N += 1
        else:
            M += 1

    while M * N > total_blocks:
        if bounding_width > bounding_height:
            N -= 1
        else:
            M -= 1

    return M, N


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


def _get_lowest_pixel_value(
    raster_path: str,
    ):
    """
    Returns the lowest valid pixel value from the first band of a raster.

    Args:
        raster_path (str): Path to the input raster file.

    Returns:
        float: Minimum non-NaN pixel value in the first band.
    """

    with rasterio.open(raster_path) as src:
        # Read the first band as a NumPy array
        data = src.read(1)
        # Replace nodata values with NaN
        nodata_value = src.nodata
        if nodata_value is not None:
            data = np.where(data == nodata_value, np.nan, data)
        # Get the minimum value, ignoring NaNs
        return np.nanmin(data)


def _add_value_to_raster(
    input_image_path: str,
    output_image_path: str,
    value: int | float,
    ):
    """
    Adds a constant value to all valid pixels in a raster and saves the result to a new file.

    Args:
        input_image_path (str): Path to the input raster file.
        output_image_path (str): Path to save the modified output raster.
        value (int | float): Value to add to each non-nodata pixel.

    Returns:
        None
    """

    # Open the source raster
    with rasterio.open(input_image_path) as src:
        # Read data as a masked array: nodata pixels are masked
        data = src.read(masked=True)

        # Retrieve the nodata value and data type
        nodata_value = src.nodata
        raster_dtype = src.dtypes[0]  # Data type of the first band

        # Ensure the value is cast to the same type as the raster data
        value = np.array(value, dtype=raster_dtype)

        # Apply the value addition only to valid (non-masked) pixels
        data += value  # Offset only the valid pixels

        # Prepare metadata for output
        out_meta = src.meta.copy()

        # Use the original data type in the output metadata
        out_meta.update({"nodata": nodata_value, "dtype": raster_dtype})

        # Fill the masked areas with the original nodata value before writing
        out_data = data.filled(nodata_value)

        # Write out the modified raster
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        with rasterio.open(output_image_path, "w", **out_meta) as dst:
            dst.write(out_data)


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


def _init_worker(img_path: str):
    """
    Initializes a global dataset cache for a worker process by opening a raster file.

    Args:
        img_path (str): Path to the image file to be opened and cached.

    Returns:
        None
    """

    import rasterio
    global _worker_dataset_cache
    _worker_dataset_cache["ds"] = rasterio.open(img_path, "r")