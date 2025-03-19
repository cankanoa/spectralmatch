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


def _get_bounding_rectangle(
    image_paths: List[str]
    ):
    """
    Gets the bounding rectangle that encompasses the bounds of all specified images.

    This function calculates a bounding rectangle by analyzing the metadata of each
    image provided in the input list. It extracts the minimum and maximum
    coordinates from the metadata and computes the smallest rectangle that
    encloses all the input images.

    Args:
        image_paths (List[str]): A list of file paths to the images for which the
            bounding rectangle is to be calculated.

    Returns:
        Tuple[float, float, float, float]: A tuple containing the minimum x
            coordinate, the minimum y coordinate, the maximum x coordinate, and
            the maximum y coordinate defining the bounding rectangle.
    """
    x_mins, y_mins, x_maxs, y_maxs = [], [], [], []

    for path in image_paths:
        transform, proj, nodata, bounds = _get_image_metadata(path)
        if bounds is not None:
            x_mins.append(bounds["x_min"])
            y_mins.append(bounds["y_min"])
            x_maxs.append(bounds["x_max"])
            y_maxs.append(bounds["y_max"])

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


def _compute_distribution_map(
    image_paths: List[str],
    bounding_rect: Tuple[float, float, float, float],
    M: int,
    N: int,
    num_bands: int,
    nodata_value: float = None,
    valid_pixel_threshold: float = 0.001,
    calculation_dtype_precision="float32",
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes a distribution map for multi-band imagery over a spatial grid.

    This function processes a list of geospatial images and computes a per-block
    mean value map and pixel count map across multiple image bands. It divides
    the area defined by the bounding rectangle into a grid with M rows and N
    columns, and aggregates pixel data within each block for each image. The
    function handles missing data through no-data values and includes logic to
    manage valid pixel thresholds.

    Args:
        image_paths (List[str]): A list of file paths to geospatial datasets to
            be processed.
        bounding_rect (Tuple[float, float, float, float]): The geographic rectangle
            bounding region of interest in the form (x_min, y_min, x_max, y_max).
        M (int): Number of rows in the grid.
        N (int): Number of columns in the grid.
        num_bands (int): The number of bands in the images to process.
        nodata_value (float, optional): The value representing invalid or missing
            data in the images. Defaults to None, which indicates no explicit
            no-data value.
        valid_pixel_threshold (float, optional): A threshold for the proportion of
            valid pixels required for a grid block to be considered valid. Defaults
            to 0.001.
        calculation_dtype_precision (str, optional): Precision of calculation arrays.
            Defaults to "float32".

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - final_block_map (np.ndarray): A 3D array of shape (M, N, num_bands)
              representing the unweighted mean value of each block across all images,
              ignoring invalid blocks.
            - final_count_map (np.ndarray): A 3D array of shape (M, N, num_bands)
              representing the total pixel count for each block across all images.

    Raises:
        None.
    """
    # Prepare accumulators
    sum_of_means_3d = np.zeros((M, N, num_bands), dtype=calculation_dtype_precision)
    image_count_3d = np.zeros((M, N, num_bands), dtype=calculation_dtype_precision)
    sum_of_counts_3d = np.zeros(
        (M, N, num_bands), dtype=calculation_dtype_precision
    )  # track total pixel counts across all images

    # Parse bounding rectangle
    x_min, y_min, x_max, y_max = bounding_rect
    block_width = (x_max - x_min) / N
    block_height = (y_max - y_min) / M

    # Process each image individually
    for path in image_paths:
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            continue

        # We'll store sums/counts for *this single image* in 3D arrays
        sum_map_3d_single = np.zeros(
            (M, N, num_bands), dtype=calculation_dtype_precision
        )
        count_map_3d_single = np.zeros(
            (M, N, num_bands), dtype=calculation_dtype_precision
        )

        for b in range(num_bands):
            # 2D accumulators for this band
            sum_map_2d = np.zeros((M, N), dtype=calculation_dtype_precision)
            count_map_2d = np.zeros((M, N), dtype=calculation_dtype_precision)

            gt = ds.GetGeoTransform()
            raster_x_size, raster_y_size = ds.RasterXSize, ds.RasterYSize

            col_indices = np.arange(raster_x_size) + 0.5
            row_indices = np.arange(raster_y_size) + 0.5

            X_geo = gt[0] + col_indices * gt[1]
            del col_indices
            gc.collect()
            Y_geo = gt[3] + row_indices * gt[5]
            del row_indices
            gc.collect()

            X_geo_2d, Y_geo_2d = np.meshgrid(X_geo, Y_geo)

            band_obj = ds.GetRasterBand(b + 1)
            arr = band_obj.ReadAsArray().astype(calculation_dtype_precision)

            # Build a valid mask
            if nodata_value is not None:
                valid_mask = arr != nodata_value
            else:
                valid_mask = np.ones_like(arr, dtype=bool)

            # Identify valid pixels
            valid_indices = np.where(valid_mask)
            del valid_mask
            gc.collect()
            pixel_values = arr[valid_indices]
            pixel_x = X_geo_2d[valid_indices]
            pixel_y = Y_geo_2d[valid_indices]
            del valid_indices
            gc.collect()

            # Determine the block indices for each valid pixel
            block_cols = np.clip(
                ((pixel_x - x_min) / block_width).astype(int), 0, N - 1
            )
            del pixel_x
            gc.collect()
            block_rows = np.clip(
                ((y_max - pixel_y) / block_height).astype(int), 0, M - 1
            )
            del pixel_y
            gc.collect()

            # Accumulate pixel sums/counts for this band
            np.add.at(sum_map_2d, (block_rows, block_cols), pixel_values)
            del pixel_values
            gc.collect()
            np.add.at(count_map_2d, (block_rows, block_cols), 1)
            del block_cols, block_rows
            gc.collect()

            # Per-block validity threshold
            # valid_blocks = count_map_2d > (valid_pixel_threshold * block_width * block_height)

            # Store sums and counts for this band in 3D arrays
            sum_map_3d_single[..., b] = sum_map_2d
            del sum_map_2d
            gc.collect()
            count_map_3d_single[..., b] = count_map_2d
            del count_map_2d
            gc.collect()

        ds = None  # done reading this image

        # Now convert each block to a *per-image* mean, ignoring invalid blocks
        block_map_3d_single = np.full(
            (M, N, num_bands), np.nan, dtype=calculation_dtype_precision
        )

        for b in range(num_bands):
            count_band = count_map_3d_single[..., b]
            sum_band = sum_map_3d_single[..., b]
            valid_blocks_band = count_band > (
                valid_pixel_threshold * block_width * block_height
            )

            block_map_3d_single[valid_blocks_band, b] = (
                sum_band[valid_blocks_band] / count_band[valid_blocks_band]
            )
            del count_band, sum_band, valid_blocks_band
            gc.collect()

        del sum_map_3d_single
        gc.collect()
        # Unweighted accumulation of these means
        #   1) For each block, if block_map_3d_single is valid, we add it to sum_of_means_3d
        #   2) We increment image_count_3d by 1 for that block (where it's valid)
        # Meanwhile, we also accumulate the pixel counts purely for reference
        valid_mask_3d = ~np.isnan(block_map_3d_single)
        sum_of_means_3d[valid_mask_3d] += block_map_3d_single[valid_mask_3d]
        del block_map_3d_single
        gc.collect()
        image_count_3d[valid_mask_3d] += 1
        del valid_mask_3d
        gc.collect()

        # Also accumulate pixel counts for reference
        # i.e., add the per-image counts so we know total coverage
        sum_of_counts_3d += count_map_3d_single

        del count_map_3d_single
        gc.collect()

    # Now compute the final unweighted average across images
    final_block_map = np.full(
        (M, N, num_bands), np.nan, dtype=calculation_dtype_precision
    )
    positive_mask = image_count_3d > 0
    final_block_map[positive_mask] = (
        sum_of_means_3d[positive_mask] / image_count_3d[positive_mask]
    )
    del sum_of_means_3d, image_count_3d, positive_mask
    gc.collect()

    # final_count_map is the sum of pixel counts from all images
    final_count_map = sum_of_counts_3d
    del sum_of_counts_3d
    gc.collect()

    return final_block_map, final_count_map


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
    projection: str = "EPSG:4326",  # Default to WGS84
    dtype="float32",  # GDAL data type as a string
    nodata_value: Optional[float] = None,
    ):
    """
    Downloads and converts a block map (numpy array) into a georeferenced raster file.

    This function creates a GeoTIFF raster file from a block map (numpy array) with
    georeferencing information based on a bounding rectangle and projection. It
    handles optional parameters for the data type and a nodata value, ensuring the
    output directory exists before writing the file.

    Args:
        block_map (np.ndarray): Numpy array containing the block data. Supports
            2D inputs (single-band images) and 3D inputs (multi-band images).
        bounding_rect (Tuple[float, float, float, float]): Tuple specifying the
            geographic extent of the data in the format (x_min, y_min, x_max, y_max).
        output_image_path (str): Path where the GeoTIFF raster file will be saved.
        projection (str): Coordinate reference system (CRS) in EPSG format. Defaults
            to "EPSG:4326" (WGS84).
        dtype (str): GDAL-compatible data type as string, such as "float32" or
            "int16". Defaults to "float32".
        nodata_value (Optional[float]): Optional value representing NoData in the
            output raster. If not provided, NoData is not set.

    Raises:
        RuntimeError: If the output dataset cannot be created.
        ValueError: If the provided projection is invalid.
    """
    output_dir = os.path.dirname(output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x_min, y_min, x_max, y_max = bounding_rect

    # Dimensions of block_map
    M, N = block_map.shape[:2]
    num_bands = 1 if len(block_map.shape) == 2 else block_map.shape[2]

    # Compute the georeferencing transform based on bounding rectangle size
    # and the number of blocks. Each block is mapped to one pixel.
    pixel_width = (x_max - x_min) / N
    pixel_height = (y_max - y_min) / M

    # Create the output dataset
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_image_path,
        N,  # Raster width  (same as number of blocks along x dimension)
        M,  # Raster height (same as number of blocks along y dimension)
        num_bands,
        gdal.GetDataTypeByName(dtype),
    )
    if not out_ds:
        raise RuntimeError(f"Could not create output dataset at {output_image_path}")

    # Assign georeferencing
    transform = (x_min, pixel_width, 0, y_max, 0, -pixel_height)
    out_ds.SetGeoTransform(transform)

    # Assign the projection
    srs = gdal.osr.SpatialReference()
    if srs.SetFromUserInput(projection) != 0:
        raise ValueError(f"Invalid projection: {projection}")
    out_ds.SetProjection(srs.ExportToWkt())

    # Write the block_map data into the output bands
    for b in range(num_bands):
        band_data = block_map if num_bands == 1 else block_map[:, :, b]
        out_band = out_ds.GetRasterBand(b + 1)

        out_band.WriteArray(band_data)

        # Optionally set NoData value
        if nodata_value is not None:
            out_band.SetNoDataValue(nodata_value)

    # Close and flush
    out_ds.FlushCache()
    out_ds = None

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
