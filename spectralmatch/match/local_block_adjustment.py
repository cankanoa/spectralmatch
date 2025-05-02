import multiprocessing as mp
import math
import gc
import os
import numpy as np
import rasterio

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
    ):

    """
    Matches histograms of input raster images using local histogram matching approach.
    This function processes raster images, adjusts their histograms locally based on
    reference blocks, and saves the corrected images to the specified output directory.
    The procedure operates block-wise on the raster images for local corrections and
    supports parallel execution for performance optimization.

    Args:
    input_image_paths (List[str]): A list of paths to input raster image files.
    output_image_folder (str): Directory path where the output images will be saved.
    output_local_basename (str): Suffix for the output filenames indicating local
    histogram matching, default is "_local".
    custom_nodata_value (float | None): Custom value to represent no data areas;
    if None, it is auto-detected from input rasters.
    target_blocks_per_image (int): Approximate number of blocks to divide each
    raster for local histogram matching, default is 100.
    alpha (float): Scaling factor for adjustment when applying histogram corrections,
    default is 1.0 (no scaling).
    calculation_dtype_precision (str): The data type precision used internally
    for corrections, default is "float32".
    output_dtype (str): Data type of the output images, default is "float32".
    projection (str): Coordinate reference system for the output rasters,
    default is "EPSG:4326".
    debug_mode (bool): Flag to enable saving intermediate block map for debugging,
    default is False.
    tile_width_and_height_tuple (Optional[Tuple[int, int]]): Optional tuple
    specifying the width and height of tiles for processing input raster,
    default is None.
    correction_method (Literal["gamma", "linear"]): Method for histogram correction,
    either "gamma" or "linear." Default is "gamma".
    parallel (bool): If True, enables parallel processing for higher efficiency,
    default is False.
    max_workers (int | None): Limits the number of parallel workers, default is
    None (uses system CPU count if parallel is True).

    Returns:
    List[str]: List of file paths to the output raster images that have been
    locally histogram-matched.
    """
    print("Start local matching")
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
                                    )
                        for b in range(num_bands)
                        for w in windows
                    ]
                    for fut in as_completed(futures):
                        win, b_idx, buf = fut.result()
                        dst.write(buf.astype(output_dtype), b_idx + 1, window=win)
                    pool.shutdown()
                else:
                    _init_worker(img_path)

                    for b in range(num_bands):
                        for win in windows:
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
                                debug_mode
                            )
                            dst.write(buf.astype(output_dtype), b_idx + 1, window=win_)
    print("Finished local matching")
    return out_paths


def _compute_tile_local(
    window: Window,
    band_idx: int,
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
    ):

    if debug_mode: print(f"Processing band: {band_idx}")

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


def _get_bounding_rectangle(
    image_paths: List[str]
    ) -> Tuple[float, float, float, float]:

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
    Computes an adjusted block size for image processing based on the coefficient
    of variation of pixel values across multiple images.

    This function calculates the coefficient of variation (CAtar) of pixel values
    from a list of images. It then compares this computed value against a
    reference coefficient, adjusting the base block size proportionally. The
    resulting block size can be used for tasks such as spatial data processing.

    Args:
    image_paths (List[str]): List of file paths to the input images.
    nodata_value (float): Value that indicates no data in the raster datasets.
    Pixels with this value will be ignored during calculations.
    reference_std (float): Reference standard deviation used for the
    coefficient of variation comparison. Default is 45.0.
    reference_mean (float): Reference mean used for the coefficient of
    variation comparison. Default is 125.0.
    base_block_size (Tuple[int, int]): Initial block size as a tuple
    (rows, columns). Default is (10, 10).
    band_index (int): Index of the image band to extract pixel values from.
    Band indexing is 1-based. Default is 1.
    calculation_dtype_precision (str): Data type to use for calculations.
    This is used for converting the raster arrays during processing.
    Default is 'float32'.

    Returns:
    Tuple[int, int]: Adjusted block size as a tuple (rows, columns). The
    block size is computed based on the ratio of computed and reference
    coefficients of variation.

    Raises:
    None
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
    C_B,
    x_frac,
    y_frac
    ):
    """
    Performs weighted bilinear interpolation on a 2D array with optional handling of NaNs.

    This function applies bilinear interpolation to the input 2D array `C_B` while properly
    handling NaN values by treating their contributions as missing during the interpolation.
    The interpolation utilizes fractional indices provided in `x_frac` and `y_frac`.

    Args:
    C_B (np.ndarray): A 2D array of numerical values, which may include NaN values.
    x_frac (np.ndarray): Fractional indices along the x-axis where interpolation is to
    be performed.
    y_frac (np.ndarray): Fractional indices along the y-axis where interpolation is to
    be performed.

    Returns:
    np.ndarray: A 1D array of interpolated values corresponding to the provided
    fractional indices. Output values are NaN where interpolation is not possible
    due to invalid neighbors.
    """

    # 1) Create a mask that is 1 where valid (not NaN), 0 where NaN
    mask = ~np.isnan(C_B)

    # 2) Replace NaNs with 0 in the original data (just for the interpolation step)
    C_B_filled = np.where(mask, C_B, 0)

    # 3) Interpolate the "filled" data
    interpolated_data = map_coordinates(
        C_B_filled, [y_frac, x_frac], order=1, mode="reflect"  # bilinear interpolation
    )

    # 4) Interpolate the mask in the same way
    interpolated_mask = map_coordinates(
        mask.astype(float), [y_frac, x_frac], order=1, mode="reflect"
    )

    # 5) Divide data by mask to get final result
    #    (this effectively averages only the non-NaN contributions)
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
                print(f"Bands added to existing raster {base}")
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
    input_image_array_path,
    target_blocks_per_image,
    bounding_rect
    ):
    """
    Calculates the optimal block size (M, N) for dividing a set of input images into
    blocks, while maintaining the aspect ratio defined by the provided bounding
    rectangle and ensuring the total number of blocks is approximately equal to the
    target.

    The function adjusts the number of rows (M) and columns (N) to fit the target
    number of blocks by scaling to the aspect ratio of the bounding rectangle.

    Args:
    input_image_array_path (list[str]): List of paths to input image arrays.
    target_blocks_per_image (int): Desired number of blocks per image.
    bounding_rect (tuple[int, int, int, int]): Tuple containing the minimum and
    maximum x and y coordinates of the bounding rectangle in the form
    (x_min, y_min, x_max, y_max).

    Returns:
    tuple[int, int]: A tuple containing the number of rows (M) and columns (N)
    that together divide the images into approximately the desired number of
    blocks, while maintaining the aspect ratio of the bounding rectangle.

    Raises:
    ValueError: If the target_blocks_per_image is less than or equal to zero.
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
    Applies gamma correction to an input array based on provided reference and input
    values, with an optional scaling factor.

    This function calculates gamma values as the ratio of the logarithm of reference values
    to the logarithm of input values and applies gamma correction to the input array
    using the formula: `P_res = alpha * P_in^gamma`.

    Args:
    arr_in (np.ndarray): The input array to be gamma-corrected.
    Mrefs (np.ndarray): Reference values used for computing gamma values.
    Mins (np.ndarray): Input values used for computing gamma values, must be greater
    than zero to avoid invalid logarithmic operations.
    alpha (float): Optional scaling factor applied during the gamma correction. Default
    is 1.0.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing:
    - Corrected output array (`arr_out_valid`) after applying gamma correction.
    - Array of gamma values computed from the reference and input values.

    Raises:
    ValueError: If any element of `Mins` is less than or equal to zero, which would
    result in invalid logarithmic operations.
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
    raster_path
    ):
    """
    Retrieves the lowest pixel value from a raster file, ignoring nodata values if they are present. This function
    reads the raster file, processes the data to handle nodata values by replacing them with NaN, and calculates
    the minimum pixel value while ignoring NaN values. It ensures compatibility with raster data formats and
    handles missing data gracefully.

    Args:
    raster_path (str): Path to the raster file to be processed. Should be a readable file compatible with
    rasterio.

    Returns:
    float: The lowest valid pixel value in the raster, ignoring nodata values. If all pixels are nodata,
    returns NaN.
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
    input_image_path,
    output_image_path,
    value
    ):
    """
    Adds a specified numeric value to all valid (non-masked) pixels in a raster file
    and writes the modified raster to a new output file.

    This function reads a raster image from the input file, modifies its data by
    adding a given value to each valid (non-nodata) pixel, and saves the result
    with the same metadata as the original raster into a specified output file.

    Args:
    input_image_path (str): Path to the input raster file.
    output_image_path (str): Path where the modified raster will be saved.
    value (float | int): Numeric value to be added to the raster data.

    Raises:
    FileNotFoundError: If the input raster file does not exist.
    RasterioIOError: If there is an issue reading or writing the raster file.
    ValueError: If the data type of the value is incompatible with the raster
    data's type.
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
    Smooths a 2D array using Gaussian filtering while handling nodata values.

    This function applies Gaussian smoothing to a 2D input array while preserving nodata
    values. It ensures that nodata regions do not influence the smoothing of valid data
    and reintroduces the nodata value in the output for consistency with the input.

    Args:
    input_array (np.ndarray): The 2D array to be smoothed.
    nodata_value (Optional[float]): The value representing missing data in
    the array. If provided, these values will be excluded from smoothing.
    Default is None.
    scale_factor (float): The smoothing scale factor (sigma) for the Gaussian
    filter. Default is 1.0.

    Returns:
    np.ndarray: The smoothed array, with nodata values reintroduced if specified.
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
    import rasterio
    global _worker_dataset_cache
    _worker_dataset_cache["ds"] = rasterio.open(img_path, "r")