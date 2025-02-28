import os
import numpy as np
import math
import rasterio
import gc

from osgeo import gdal
from typing import Tuple, List, Optional
from scipy.ndimage import map_coordinates, gaussian_filter


def merge_rasters(input_array, output_image_folder, output_file_name="merge.tif"):
    output_path = os.path.join(output_image_folder, output_file_name)
    input_datasets = [gdal.Open(path) for path in input_array if gdal.Open(path)]
    gdal.Warp(
        output_path,
        input_datasets,
        format='GTiff',
    )
    print(f"Merged raster saved to: {output_path}")

def get_image_metadata(input_image_path):
    """
Get metadata of a TIFF image, including transform, projection, nodata, and bounds.
    """
    try:
        dataset = gdal.Open(input_image_path, gdal.GA_ReadOnly)
        if dataset is not None:
            transform = dataset.GetGeoTransform()
            projection = dataset.GetProjection()
            nodata = None
            if dataset.RasterCount > 0:
                nodata = dataset.GetRasterBand(1).GetNoDataValue()

            if transform:
                x_min = transform[0]
                y_max = transform[3]
                x_max = x_min + (dataset.RasterXSize * transform[1])
                y_min = y_max + (dataset.RasterYSize * transform[5])
                bounds = {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
            else:
                bounds = None

            dataset = None
            return transform, projection, nodata, bounds
        else:
            print(f"Could not open the file: {input_image_path}")
    except Exception as e:
        print(f"Error processing {input_image_path}: {e}")
    return None, None, None, None

def get_bounding_rectangle(image_paths: List[str]):
    """
Computes the minimum bounding rectangle (x_min, y_min, x_max, y_max)
that covers all input images.
    """
    x_mins, y_mins, x_maxs, y_maxs = [], [], [], []

    for path in image_paths:
        transform, proj, nodata, bounds = get_image_metadata(path)
        if bounds is not None:
            x_mins.append(bounds["x_min"])
            y_mins.append(bounds["y_min"])
            x_maxs.append(bounds["x_max"])
            y_maxs.append(bounds["y_max"])

    return (min(x_mins), min(y_mins), max(x_maxs), max(y_maxs))


def compute_mosaic_coefficient_of_variation(
        image_paths: List[str],
        nodata_value: float,
        reference_std: float = 45.0,
        reference_mean: float = 125.0,
        base_block_size: Tuple[int, int] = (10, 10),
        band_index: int = 1,
        calculation_dtype_precision = 'float32'
) -> Tuple[int, int]:
    all_pixels = []

    # Collect pixel values from all images
    for path in image_paths:
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            continue
        band = ds.GetRasterBand(band_index)
        arr = band.ReadAsArray().astype(calculation_dtype_precision)
        if nodata_value is not None:
            mask = (arr != nodata_value)
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


def compute_distribution_map(
        image_paths: List[str],
        bounding_rect: Tuple[float, float, float, float],
        M: int,
        N: int,
        num_bands: int,
        nodata_value: float = None,
        valid_pixel_threshold: float = 0.001,
        calculation_dtype_precision = 'float32'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divides the bounding rectangle into (M x N) blocks and performs
    a "mean of the means" across all images (unweighted).

    For each image:
    - Compute per-block sums and counts.
    - Convert to a block mean for that image.
    Then combine those image-level block means with a simple average
    (each image is 'one vote'), ignoring blocks that are invalid (NaN)
    in a particular image.

    Returns
    -------
    final_block_map : np.ndarray
    Shape (M, N, num_bands). The unweighted average of each image's
    block means, per band. If a block is invalid in all images,
    it remains NaN.
    final_count_map : np.ndarray
    Shape (M, N, num_bands). The sum of pixel counts from *all* images
    for each block/band (not used to weight the mean, purely informational).
    """
    # Prepare accumulators
    sum_of_means_3d = np.zeros((M, N, num_bands), dtype=calculation_dtype_precision)
    image_count_3d = np.zeros((M, N, num_bands), dtype=calculation_dtype_precision)
    sum_of_counts_3d = np.zeros((M, N, num_bands), dtype=calculation_dtype_precision)  # track total pixel counts across all images

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
        sum_map_3d_single = np.zeros((M, N, num_bands), dtype=calculation_dtype_precision)
        count_map_3d_single = np.zeros((M, N, num_bands), dtype=calculation_dtype_precision)

        for b in range(num_bands):
            # 2D accumulators for this band
            sum_map_2d = np.zeros((M, N), dtype=calculation_dtype_precision)
            count_map_2d = np.zeros((M, N), dtype=calculation_dtype_precision)

            gt = ds.GetGeoTransform()
            raster_x_size, raster_y_size = ds.RasterXSize, ds.RasterYSize

            col_indices = np.arange(raster_x_size) + 0.5
            row_indices = np.arange(raster_y_size) + 0.5

            X_geo = gt[0] + col_indices * gt[1]
            del col_indices; gc.collect()
            Y_geo = gt[3] + row_indices * gt[5]
            del row_indices; gc.collect()

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
            del valid_mask; gc.collect()
            pixel_values = arr[valid_indices]
            pixel_x = X_geo_2d[valid_indices]
            pixel_y = Y_geo_2d[valid_indices]
            del valid_indices; gc.collect()

            # Determine the block indices for each valid pixel
            block_cols = np.clip(((pixel_x - x_min) / block_width).astype(int), 0, N - 1)
            del pixel_x; gc.collect()
            block_rows = np.clip(((y_max - pixel_y) / block_height).astype(int), 0, M - 1)
            del pixel_y; gc.collect()

            # Accumulate pixel sums/counts for this band
            np.add.at(sum_map_2d, (block_rows, block_cols), pixel_values)
            del pixel_values; gc.collect()
            np.add.at(count_map_2d, (block_rows, block_cols), 1)
            del block_cols, block_rows; gc.collect()

            # Per-block validity threshold
            # valid_blocks = count_map_2d > (valid_pixel_threshold * block_width * block_height)

            # Store sums and counts for this band in 3D arrays
            sum_map_3d_single[..., b] = sum_map_2d
            del sum_map_2d; gc.collect()
            count_map_3d_single[..., b] = count_map_2d
            del count_map_2d; gc.collect()

        ds = None  # done reading this image

        # Now convert each block to a *per-image* mean, ignoring invalid blocks
        block_map_3d_single = np.full((M, N, num_bands), np.nan, dtype=calculation_dtype_precision)

        for b in range(num_bands):
            count_band = count_map_3d_single[..., b]
            sum_band = sum_map_3d_single[..., b]
            valid_blocks_band = count_band > (valid_pixel_threshold * block_width * block_height)

            block_map_3d_single[valid_blocks_band, b] = (
                sum_band[valid_blocks_band] / count_band[valid_blocks_band]
            )
            del count_band, sum_band, valid_blocks_band; gc.collect()

        del sum_map_3d_single; gc.collect()
        # Unweighted accumulation of these means
        #   1) For each block, if block_map_3d_single is valid, we add it to sum_of_means_3d
        #   2) We increment image_count_3d by 1 for that block (where it's valid)
        # Meanwhile, we also accumulate the pixel counts purely for reference
        valid_mask_3d = ~np.isnan(block_map_3d_single)
        sum_of_means_3d[valid_mask_3d] += block_map_3d_single[valid_mask_3d]
        del block_map_3d_single; gc.collect()
        image_count_3d[valid_mask_3d] += 1
        del valid_mask_3d; gc.collect()

        # Also accumulate pixel counts for reference
        # i.e., add the per-image counts so we know total coverage
        sum_of_counts_3d += count_map_3d_single

        del count_map_3d_single; gc.collect()

    # Now compute the final unweighted average across images
    final_block_map = np.full((M, N, num_bands), np.nan, dtype=calculation_dtype_precision)
    positive_mask = image_count_3d > 0
    final_block_map[positive_mask] = sum_of_means_3d[positive_mask] / image_count_3d[positive_mask]
    del sum_of_means_3d, image_count_3d, positive_mask; gc.collect()

    # final_count_map is the sum of pixel counts from all images
    final_count_map = sum_of_counts_3d
    del sum_of_counts_3d; gc.collect()

    return final_block_map, final_count_map


def weighted_bilinear_interpolation(C_B, x_frac, y_frac):
    # 1) Create a mask that is 1 where valid (not NaN), 0 where NaN
    mask = ~np.isnan(C_B)

    # 2) Replace NaNs with 0 in the original data (just for the interpolation step)
    C_B_filled = np.where(mask, C_B, 0)

    # 3) Interpolate the "filled" data
    interpolated_data = map_coordinates(
        C_B_filled,
        [y_frac, x_frac],
        order=1,          # bilinear interpolation
        mode='reflect'
    )

    # 4) Interpolate the mask in the same way
    interpolated_mask = map_coordinates(
        mask.astype(float),
        [y_frac, x_frac],
        order=1,
        mode='reflect'
    )

    # 5) Divide data by mask to get final result
    #    (this effectively averages only the non-NaN contributions)
    with np.errstate(divide='ignore', invalid='ignore'):
        interpolated_values = interpolated_data / interpolated_mask
        # Where the interpolated mask is 0, set output to NaN (no valid neighbors)
        interpolated_values[interpolated_mask == 0] = np.nan

    return interpolated_values


def download_block_map(
        block_map: np.ndarray,
        bounding_rect: Tuple[float, float, float, float],
        output_image_path: str,
        projection: str = "EPSG:4326",  # Default to WGS84
        dtype = 'float32',         # GDAL data type as a string
        nodata_value: Optional[float] = None
):
    """
    Export the block_map as a georeferenced raster image.

    Parameters:
    block_map (np.ndarray): Block map (M x N or M x N x num_bands) to export.
    bounding_rect (tuple): Bounding rectangle (x_min, y_min, x_max, y_max).
    output_image_path (str): Path to save the output raster image.
    projection (str): Coordinate system for the raster (e.g., EPSG:4326, EPSG:3857).
    dtype (str): GDAL-compatible data type (e.g., Float32, Int16).
    nodata_value (float): Value to assign to NoData pixels.
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
        gdal.GetDataTypeByName(dtype)
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


def compute_block_size(input_image_array_path, target_blocks_per_image, bounding_rect):
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

def apply_gamma_correction(
        arr_in: np.ndarray,
        Mrefs: np.ndarray,
        Mins: np.ndarray,
        alpha: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
Perform gamma correction using the combined equation for valid pixels.

Parameters:
arr_in_valid (np.ndarray): Array of valid input pixel values (P_in(x, y)).
Mrefs_valid (np.ndarray): Array of reference mean values (M_ref(x, y)).
Mins_valid (np.ndarray): Array of input mean values (M_in(x, y)).
alpha (float): Scaling constant (default is 1.0).

Returns:
np.ndarray: Gamma-corrected pixel values (P_res(x, y)).
    """
    # Ensure no division by zero
    if np.any(Mins <= 0):
        raise ValueError("Mins_valid contains zero or negative values, which are invalid for logarithmic operations. Maybe offset isnt working correctly.")

    # Compute gamma values (log(M_ref) / log(M_in))
    gammas = np.log(Mrefs) / np.log(Mins)

    # Apply gamma correction: P_res = alpha * P_in^gamma
    arr_out_valid = alpha * (arr_in ** gammas)


    return arr_out_valid, gammas


def get_lowest_pixel_value(raster_path):
    """
Get the lowest pixel value in a single raster file.

Parameters:
raster_path (str): Path to the raster file.

Returns:
float: The lowest pixel value in the raster.
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

def add_value_to_raster(input_image_path, output_image_path, value):
    """
Opens a raster, adds 'value' only to valid pixels (excluding nodata pixels),
and writes the result to a new raster, preserving the original nodata value.
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
        out_meta.update({'nodata': nodata_value, 'dtype': raster_dtype})

        # Fill the masked areas with the original nodata value before writing
        out_data = data.filled(nodata_value)

        # Write out the modified raster
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        with rasterio.open(output_image_path, 'w', **out_meta) as dst:
            dst.write(out_data)


def smooth_array(input_array: np.ndarray, nodata_value: Optional[float] = None, scale_factor: float = 1.0) -> np.ndarray:
    """
Smooth a NumPy array using Gaussian smoothing while excluding NaN or NoData values.

Parameters:
input_array (np.ndarray): The input array to smooth.
nodata_value (float, optional): The value representing NoData. Regions with this value are excluded from smoothing.
scale_factor (float): The standard deviation for Gaussian kernel. Higher values increase smoothing.

Returns:
np.ndarray: Smoothed array with nodata regions preserved.
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
    smoothed_normalized = np.where(normalization_mask > 0, smoothed / normalization_mask, np.nan)

    # Reapply the nodata value (if specified) for output consistency
    if nodata_value is not None:
        smoothed_normalized = np.where(valid_mask, smoothed_normalized, nodata_value)

    return smoothed_normalized


def process_local_histogram_matching(
        input_image_paths: List[str],
        output_image_folder: str,
        output_local_basename: str,
        global_nodata_value: float = -9999,
        target_blocks_per_image: int = 100,
        alpha: float = 1.0,
        calculation_dtype_precision = 'float32',
        floor_value: Optional[float] = None,
        gamma_bounds: Optional[Tuple[float, float]] = None
):

    # Its better to compute this offset right before gamma correciton, apply, then reverse
    # print('-------------------- Computing offset to make raster pixels > 0')
    # lowest_value: float = None
    # pixels_positive_offset: int = None
    # offset_image_paths = []
    #
    # # Find the lowest pixel value across all input rasters
    # for raster_path in input_image_paths:
    #     value = get_lowest_pixel_value(raster_path)
    #     if lowest_value is None or value < lowest_value:
    #         lowest_value = value
    # print(f'Lowest_value: {lowest_value}')
    #
    # if lowest_value <= 0:
    #     pixels_positive_offset = int(abs(lowest_value))+1
    #     for raster_path in input_image_paths:
    #
    #         offset_image_path = os.path.join(output_image_folder,"OffsetImages",f"{os.path.splitext(os.path.basename(raster_path))[0]}_OffsetImage{os.path.splitext(raster_path)[1]}")
    #
    #         offset_image_paths.append(offset_image_path)
    #
    #         add_value_to_raster(raster_path, offset_image_path, pixels_positive_offset)
    #         print(f"Offset of {pixels_positive_offset} saved: {offset_image_path}")
    # input_image_paths = offset_image_paths

    print('-------------------- Computing block size')
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder, exist_ok=True)

    # bounding rectangle
    bounding_rect = get_bounding_rectangle(input_image_paths)
    print(f"Bounding rectangle: {bounding_rect}")

    M, N = compute_block_size(input_image_paths, target_blocks_per_image, bounding_rect)
    print(f'Blocks(M,N): {M}, {N} = {M * N}')

    # -- Buffer boundary
    # print(f"Adjusted bounding rectangle (with 0.5-block offset): {bounding_rect}")
    # x_min, y_min, x_max, y_max = bounding_rect
    #
    # # compute block_width, block_height *before* expanding
    # block_width = (x_max - x_min) / N
    # block_height = (y_max - y_min) / M
    #
    # # Expand by half a block on each side
    # x_min -= block_width * 0.5
    # x_max += block_width * 0.5
    # y_min -= block_height * 0.5
    # y_max += block_height * 0.5
    #
    # # Re-assign the adjusted bounding_rect
    # bounding_rect = (x_min, y_min, x_max, y_max)

    # -- Alternatie aproach from the paper which I dont like
    # M, N = compute_mosaic_coefficient_of_variation(input_image_paths, global_nodata_value)

    print('-------------------- Computing global reference block map')
    num_bands = gdal.Open(input_image_paths[0], gdal.GA_ReadOnly).RasterCount

    ref_map, ref_count_map = compute_distribution_map(
        input_image_paths,
        bounding_rect,
        M,
        N,
        num_bands,
        nodata_value=global_nodata_value
    )

    # ref_map = smooth_array(ref_map, nodata_value=global_nodata_value)

    download_block_map(
        block_map=np.nan_to_num(ref_map, nan=global_nodata_value),
        bounding_rect=bounding_rect,
        output_image_path=os.path.join(output_image_folder, "RefDistMap.tif"),
        nodata_value=global_nodata_value,
        projection = "EPSG:6635",
    )

    corrected_paths = []
    for img_path in input_image_paths:
        print(f'-------------------- Processing: {img_path}')
        print(f'-------------------- Computing local block map')
        loc_map, loc_count_map = compute_distribution_map(
            [img_path],
            bounding_rect,
            M,
            N,
            num_bands,
            nodata_value=global_nodata_value
        )

        # loc_map = smooth_array(loc_map, nodata_value=global_nodata_value)

        out_name = os.path.splitext(os.path.basename(img_path))[0] + output_local_basename + ".tif"
        out_path = os.path.join(output_image_folder, out_name)

        download_block_map(
            block_map=np.nan_to_num(loc_count_map, nan=global_nodata_value),
            bounding_rect=bounding_rect,
            output_image_path=os.path.join(
                os.path.dirname(out_path),
                "locCountMaps",
                os.path.splitext(os.path.basename(out_path))[0] + "_locCountMaps" + os.path.splitext(out_path)[1]),
            nodata_value=global_nodata_value,
            projection = "EPSG:6635",
        )

        download_block_map(
            block_map=np.nan_to_num(loc_map, nan=global_nodata_value),
            bounding_rect=bounding_rect,
            output_image_path=os.path.join(
                os.path.dirname(out_path),
                "LocalDistMaps",
                os.path.splitext(os.path.basename(out_path))[0] + "_LocalDistMap" + os.path.splitext(out_path)[1]),
            nodata_value=global_nodata_value,
            projection = "EPSG:6635",
        )

        print(f'-------------------- Computing local correction, applying, and saving')
        ds_in = gdal.Open(img_path, gdal.GA_ReadOnly) or RuntimeError(f"Could not open {img_path}")
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(out_path, ds_in.RasterXSize, ds_in.RasterYSize, num_bands, gdal.GetDataTypeByName(calculation_dtype_precision))

        gt = ds_in.GetGeoTransform()
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(ds_in.GetProjection())

        x_min, y_max = gt[0], gt[3]
        x_max, y_min = x_min + gt[1] * ds_in.RasterXSize, y_max + gt[5] * ds_in.RasterYSize
        this_image_bounds = (x_min, y_min, x_max, y_max)

        for b in range(num_bands):
            print(f'-------------------- For band {b + 1}')
            band_in = ds_in.GetRasterBand(b + 1)
            arr_in = band_in.ReadAsArray().astype(calculation_dtype_precision)

            # Generate a single matrix for block indices directly
            nX, nY = ds_in.RasterXSize, ds_in.RasterYSize
            col_index = np.arange(nX)
            row_index = np.arange(nY)
            Xgeo = gt[0] + (col_index * gt[1])
            Ygeo = gt[3] + (row_index * gt[5])
            Xgeo_2d, Ygeo_2d = np.meshgrid(Xgeo, Ygeo)

            # Compute block indices for each pixel
            row_fs = np.clip(((bounding_rect[3] - Ygeo_2d) / (bounding_rect[3] - bounding_rect[1])) * M - 0.5, 0, M - 1)
            col_fs = np.clip((((Xgeo_2d - bounding_rect[0]) / (bounding_rect[2] - bounding_rect[0])) * N) - 0.5, 0, N - 1)

            arr_out = np.full_like(arr_in, global_nodata_value, dtype=calculation_dtype_precision)
            valid_mask = arr_in != global_nodata_value

            # Extract the band-specific local and reference maps
            loc_band_2d = loc_map[:, :, b]
            ref_band_2d = ref_map[:, :, b]

            # Mask out regions where loc_map or ref_map are NaN (NoData regions)
            valid_loc_mask = ~np.isnan(loc_band_2d)
            valid_ref_mask = ~np.isnan(ref_band_2d)

            # Ensure valid_mask is correctly applied to the input arrays
            valid_rows, valid_cols = np.where(valid_mask)

            download_block_map(
                block_map=np.where(valid_mask, 1, global_nodata_value),
                bounding_rect=this_image_bounds,
                output_image_path=os.path.join(
                    os.path.dirname(out_path),
                    "ValidMasks",
                    os.path.splitext(os.path.basename(out_path))[0] + f"_ValidMask_{b}.tif"),
                projection="EPSG:6635",
                nodata_value=global_nodata_value
            )

            # Ensure weighted interpolation handles only valid regions
            Mrefs = np.full_like(arr_in, global_nodata_value, dtype=calculation_dtype_precision)
            Mins = np.full_like(arr_in, global_nodata_value, dtype=calculation_dtype_precision)

            Mrefs[valid_rows, valid_cols] = weighted_bilinear_interpolation(
                ref_band_2d,
                # ref_count_map[:, :, b],
                col_fs[valid_rows, valid_cols],
                row_fs[valid_rows, valid_cols]
            )
            Mins[valid_rows, valid_cols] = weighted_bilinear_interpolation(
                loc_band_2d,
                # loc_count_map[:, :, b],
                col_fs[valid_rows, valid_cols],
                row_fs[valid_rows, valid_cols]
            )

            download_block_map(
                block_map=Mrefs,
                bounding_rect=this_image_bounds,
                output_image_path=os.path.join(
                    os.path.dirname(out_path),
                    "Mrefs",
                    os.path.splitext(os.path.basename(out_path))[0] + f"_Mrefs_{b}.tif"),
                projection="EPSG:6635",
                nodata_value=global_nodata_value
            )

            download_block_map(
                block_map=Mins,
                bounding_rect=this_image_bounds,
                output_image_path=os.path.join(
                    os.path.dirname(out_path),
                    "Mins",
                    os.path.splitext(os.path.basename(out_path))[0] + f"_Mins_{b}.tif"),
                projection="EPSG:6635",
                nodata_value=global_nodata_value
            )

            valid_pixels = valid_mask #& (Mrefs > 0) & (Mins > 0) # Mask if required but better to offset values <= 0
            smallest_value = np.min([arr_in[valid_pixels], Mrefs[valid_pixels], Mins[valid_pixels]])

            if smallest_value <= 0:
                pixels_positive_offset = abs(smallest_value) + 1
                arr_out[valid_pixels], gammas = apply_gamma_correction(arr_in[valid_pixels]+pixels_positive_offset, Mrefs[valid_pixels]+pixels_positive_offset, Mins[valid_pixels]+pixels_positive_offset, alpha)
                arr_out[valid_pixels] = arr_out[valid_pixels] - pixels_positive_offset
            else:
                arr_out[valid_pixels], gammas = apply_gamma_correction(arr_in[valid_pixels], Mrefs[valid_pixels], Mins[valid_pixels], alpha)


            gammas_array = np.full(arr_in.shape, global_nodata_value, dtype=calculation_dtype_precision)
            gammas_array[valid_rows, valid_cols] = gammas
            download_block_map(
                block_map=gammas_array,
                bounding_rect=this_image_bounds,
                output_image_path=os.path.join(
                    os.path.dirname(out_path),
                    "Gammas",
                    os.path.splitext(os.path.basename(out_path))[0] + f"_Gamma_{b}.tif"
                ),
                projection="EPSG:6635",
                nodata_value=global_nodata_value
            )

            # arr_out[valid_pixels] = arr_in[valid_pixels] * (Mrefs[valid_pixels] / Mins[valid_pixels]) # An alternative way to calculate the corrected raster

            out_band = out_ds.GetRasterBand(b + 1)
            out_band.WriteArray(arr_out)
            out_band.SetNoDataValue(global_nodata_value)

        ds_in = None
        out_ds.FlushCache()
        out_ds = None

        corrected_paths.append(out_path)
        print(f"Saved: {out_path}")

    # 6) Merge final corrected rasters
    print("Merging saved rasters")
    merge_rasters(corrected_paths, output_image_folder, output_file_name=f"Merged{output_local_basename}.tif")
    print("Local histogram matching done")
    