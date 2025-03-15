import sys
import os
import numpy as np

from osgeo import gdal
from scipy.optimize import least_squares
from utils_common import _merge_rasters, _get_image_metadata

np.set_printoptions(
    suppress=True,
    precision=3,
    linewidth=300,
    formatter={"float_kind": lambda x: f"{x: .3f}"},
    )


def _find_overlaps(
    image_bounds_dict
    ):
    """
    Determines overlaps between rectangular regions defined in a dictionary of image bounds.

    Each rectangular region is defined by the minimum and maximum coordinates in both
    x and y directions. The function identifies pairs of regions that overlap with each
    other, based on their bounds. An overlap is determined if one rectangle's area
    intersects with another.

    Args:
        image_bounds_dict (dict): A dictionary where keys represent unique identifiers
                                  for rectangular regions and values are dictionaries
                                  containing the bounds of each region. The bounds
                                  dictionary must include the keys 'x_min', 'x_max',
                                  'y_min', and 'y_max'.

    Returns:
        tuple: A tuple of tuples, where each nested tuple contains two keys from the
               input dictionary representing regions that overlap with one another.
    """
    overlaps = []

    for key1, bounds1 in image_bounds_dict.items():
        for key2, bounds2 in image_bounds_dict.items():
            if key1 < key2:  # Avoid duplicate and self-comparison
                if (
                    bounds1["x_min"] < bounds2["x_max"]
                    and bounds1["x_max"] > bounds2["x_min"]
                    and bounds1["y_min"] < bounds2["y_max"]
                    and bounds1["y_max"] > bounds2["y_min"]
                ):
                    overlaps.append((key1, key2))

    return tuple(overlaps)


def _calculate_image_stats(
    num_bands,
    input_image_path_i,
    input_image_path_j,
    id_i,
    id_j,
    bound_i,
    bound_j,
    nodata_i,
    nodata_j,
    ):
    """
    Calculate overlap statistics (mean, std, and size) for two overlapping images,
    as well as their individual statistics, while excluding NoData values.

    Args:
        num_bands (int): Number of bands in the images.
        input_image_path_i (str): Path to the first image.
        input_image_path_j (str): Path to the second image.
        id_i (int): ID of the first image.
        id_j (int): ID of the second image.
        bound_i (dict): Bounds of the first image in the format {"x_min", "x_max", "y_min", "y_max"}.
        bound_j (dict): Bounds of the second image in the format {"x_min", "x_max", "y_min", "y_max"}.
        nodata_i (float): The NoData value for the first image.
        nodata_j (float): The NoData value for the second image.

    Returns:
        tuple: A tuple containing:
        - overlap_stat: Dictionary of overlap statistics in the format:
        {id_i: {id_j: {band: {'mean': value, 'std': value, 'size': value}}},
        id_j: {id_i: {band: {'mean': value, 'std': value, 'size': value}}}}
        - whole_stats: Dictionary of whole image statistics in the format:
        {id_i: {band: {'mean': value, 'std': value, 'size': value}}}
    """
    # Initialize the result dictionaries
    overlap_stat = {id_i: {id_j: {}}, id_j: {id_i: {}}}
    whole_stats = {id_i: {}, id_j: {}}

    # Open both images
    dataset_i = gdal.Open(input_image_path_i)
    dataset_j = gdal.Open(input_image_path_j)

    if not dataset_i or not dataset_j:
        raise RuntimeError("Failed to open one or both input images.")

    geo_transform_i = dataset_i.GetGeoTransform()
    geo_transform_j = dataset_j.GetGeoTransform()

    # Calculate overlap bounds
    x_min_overlap = max(bound_i["x_min"], bound_j["x_min"])
    x_max_overlap = min(bound_i["x_max"], bound_j["x_max"])
    y_min_overlap = max(bound_i["y_min"], bound_j["y_min"])
    y_max_overlap = min(bound_i["y_max"], bound_j["y_max"])

    if x_min_overlap >= x_max_overlap or y_min_overlap >= y_max_overlap:
        return overlap_stat, whole_stats

    # Calculate pixel ranges for the overlaps
    col_min_i = int((x_min_overlap - geo_transform_i[0]) / geo_transform_i[1])
    col_max_i = int((x_max_overlap - geo_transform_i[0]) / geo_transform_i[1])
    row_min_i = int((y_max_overlap - geo_transform_i[3]) / geo_transform_i[5])
    row_max_i = int((y_min_overlap - geo_transform_i[3]) / geo_transform_i[5])

    col_min_j = int((x_min_overlap - geo_transform_j[0]) / geo_transform_j[1])
    col_max_j = int((x_max_overlap - geo_transform_j[0]) / geo_transform_j[1])
    row_min_j = int((y_max_overlap - geo_transform_j[3]) / geo_transform_j[5])
    row_max_j = int((y_min_overlap - geo_transform_j[3]) / geo_transform_j[5])

    # Ensure overlap shapes align
    overlap_rows = min(row_max_i - row_min_i, row_max_j - row_min_j)
    overlap_cols = min(col_max_i - col_min_i, col_max_j - col_min_j)

    row_max_i, row_max_j = row_min_i + overlap_rows, row_min_j + overlap_rows
    col_max_i, col_max_j = col_min_i + overlap_cols, col_min_j + overlap_cols

    # Process each band
    for band_idx in range(num_bands):
        band_data_i = dataset_i.GetRasterBand(band_idx + 1).ReadAsArray()
        band_data_j = dataset_j.GetRasterBand(band_idx + 1).ReadAsArray()

        mask_i = band_data_i != nodata_i
        mask_j = band_data_j != nodata_j

        # Calculate whole image statistics for each image
        valid_data_i = band_data_i[mask_i]
        valid_data_j = band_data_j[mask_j]
        whole_stats[id_i][band_idx] = {
            "mean": np.mean(valid_data_i) if valid_data_i.size > 0 else 0,
            "std": np.std(valid_data_i) if valid_data_i.size > 0 else 0,
            "size": valid_data_i.size,
        }
        whole_stats[id_j][band_idx] = {
            "mean": np.mean(valid_data_j) if valid_data_j.size > 0 else 0,
            "std": np.std(valid_data_j) if valid_data_j.size > 0 else 0,
            "size": valid_data_j.size,
        }

        # Slice data and masks for overlaps
        overlap_data_i = band_data_i[row_min_i:row_max_i, col_min_i:col_max_i]
        overlap_data_j = band_data_j[row_min_j:row_max_j, col_min_j:col_max_j]
        overlap_mask_i = mask_i[row_min_i:row_max_i, col_min_i:col_max_i]
        overlap_mask_j = mask_j[row_min_j:row_max_j, col_min_j:col_max_j]

        # Combine masks and calculate overlap data
        overlap_mask = overlap_mask_i & overlap_mask_j
        overlap_data_combined_i = overlap_data_i[overlap_mask]
        overlap_data_combined_j = overlap_data_j[overlap_mask]

        # Calculate overlap statistics
        mean1 = (
            np.mean(overlap_data_combined_i) if overlap_data_combined_i.size > 0 else 0
        )
        std1 = (
            np.std(overlap_data_combined_i) if overlap_data_combined_i.size > 0 else 0
        )
        mean2 = (
            np.mean(overlap_data_combined_j) if overlap_data_combined_j.size > 0 else 0
        )
        std2 = (
            np.std(overlap_data_combined_j) if overlap_data_combined_j.size > 0 else 0
        )
        overlap_size = overlap_mask.sum()

        overlap_stat[id_i][id_j][band_idx] = {
            "mean": mean1,
            "std": std1,
            "size": overlap_size,
        }
        overlap_stat[id_j][id_i][band_idx] = {
            "mean": mean2,
            "std": std2,
            "size": overlap_size,
        }

    dataset_i = None
    dataset_j = None

    return overlap_stat, whole_stats


def _append_band_to_tif(
    band_array: np.ndarray,
    transform: tuple,
    projection: str,
    output_path: str,
    nodata_value: float,
    band_index: int,
    total_bands: int,
    dtype=gdal.GDT_Int16,
    ):
    """
    Writes or appends a raster band to a GeoTIFF file.

    This function is designed to either create a new GeoTIFF file with specific
    georeferencing, projection, and metadata, or to append a band to an
    existing GeoTIFF file. If the file does not exist, it will be created with
    the defined number of bands (`total_bands`) where all bands will be
    initialized with a nodata value and the specified `band_array` will be
    written to the provided `band_index`. If the file already exists, only the
    specified band is updated with the data provided in `band_array`.

    Args:
        band_array (np.ndarray): 2D array representing the raster data to
            write or append as a band in the GeoTIFF.
        transform (tuple): Affine transformation coefficients to georeference
            the raster data in the GeoTIFF.
        projection (str): Spatial reference system (e.g., WKT format) to
            assign to the output GeoTIFF.
        output_path (str): File path to create or update the GeoTIFF file.
        nodata_value (float): Value to represent nodata in the raster bands
            of the GeoTIFF.
        band_index (int): One-based index specifying which band to write the
            raster data.
        total_bands (int): Total number of bands to allocate when creating the
            GeoTIFF file (ignored if the file already exists).
        dtype: GDAL data type (default is `gdal.GDT_Int16`) to define the pixel type
            of all raster bands in the GeoTIFF. Default is `gdal.GDT_Int16`.

    Raises:
        RuntimeError: If the function fails to open an existing GeoTIFF in
            update mode.
        ValueError: If the specified `band_index` exceeds the total number
            of bands available in the existing GeoTIFF.
    """
    rows, cols = band_array.shape

    # 1. If file does not exist, create it
    if not os.path.exists(output_path):
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(output_path, cols, rows, total_bands, dtype)
        # Set georeferencing only once (on creation)
        out_ds.SetGeoTransform(transform)
        out_ds.SetProjection(projection)

        # Initialize all bands to the nodata (optional, if you want them pre-filled)
        for b_i in range(1, total_bands + 1):
            band_tmp = out_ds.GetRasterBand(b_i)
            band_tmp.Fill(nodata_value)
            band_tmp.SetNoDataValue(nodata_value)

        # Now write our band
        out_band = out_ds.GetRasterBand(band_index)
        out_band.WriteArray(band_array)
        out_band.SetNoDataValue(nodata_value)

        out_ds.FlushCache()
        out_ds = None

    else:
        # 2. If file exists, open in UPDATE mode
        out_ds = gdal.Open(output_path, gdal.GA_Update)
        if out_ds is None:
            raise RuntimeError(f"Could not open {output_path} in update mode.")

        # We assume it already has the correct 'total_bands' allocated
        if band_index > out_ds.RasterCount:
            raise ValueError(
                f"Band index {band_index} exceeds "
                f"the number of bands ({out_ds.RasterCount}) in {output_path}."
            )

        # Write the band
        out_band = out_ds.GetRasterBand(band_index)
        out_band.WriteArray(band_array)
        out_band.SetNoDataValue(nodata_value)

        out_ds.FlushCache()
        out_ds = None  # close

    print(f"Appended band {band_index} to {output_path}.")


def _save_multiband_as_geotiff(
    array,
    geo_transform,
    projection,
    path,
    nodata_values
    ):
    """
    Save a multi-band array as a GeoTIFF file.

    This function takes a 3D NumPy array and saves it as a multi-band GeoTIFF file.
    The function utilizes GDAL for the creation and writing process. The GeoTIFF file
    is created according to the specified geospatial metadata, such as geo-transform
    and projection, and supports the optional assignment of no-data values for the bands.

    Args:
        array: The 3D NumPy array (dimensions: bands, rows, columns) representing
            the raster data to be saved.
        geo_transform: The geo-transform parameters used to map pixel locations
            to geographic coordinates.
        projection: The spatial reference system (projection) as a string in WKT format.
        path: The file path where the GeoTIFF will be saved.
        nodata_values: Optional; a value to set for no-data pixels in each band.
            Should be set to None if no no-data values are to be assigned.
    """
    driver = gdal.GetDriverByName("GTiff")
    num_bands, rows, cols = array.shape
    out_ds = driver.Create(path, cols, rows, num_bands, gdal.GDT_Int16)
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)

    for i in range(num_bands):
        out_band = out_ds.GetRasterBand(i + 1)
        out_band.WriteArray(array[i])
        if nodata_values is not None:
            out_band.SetNoDataValue(nodata_values)

    out_ds.FlushCache()


