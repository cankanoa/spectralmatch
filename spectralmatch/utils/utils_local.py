import os
from doctest import debug
import numpy as np
import math
import rasterio
import gc

from osgeo import gdal
from typing import Tuple, List, Optional
from scipy.ndimage import map_coordinates, gaussian_filter
from spectralmatch.utils.utils_common import _get_image_metadata
from rasterio.windows import Window
from spectralmatch.utils.utils_io import create_windows

def _get_bounding_rectangle(image_paths: List[str]) -> Tuple[float, float, float, float]:
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
                    windows = create_windows(data.width, data.height, tile_width_and_height_tuple[0], tile_width_and_height_tuple[1])
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


import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.errors import RasterioIOError
from typing import Tuple, Optional

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
                print(f"Bands added to existing raster at: {output_image_path}")
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
