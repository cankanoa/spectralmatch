import sys
import os
import numpy as np

from osgeo import gdal, ogr, osr
from scipy.optimize import least_squares

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
    vector_mask_path=None,
):
    """
    Calculate overlap statistics (mean, std, and size) for two overlapping images,
    as well as their individual statistics, while excluding NoData values. Optionally
    applies a GPKG-based vector mask if `vector_mask_path` is provided.

    If `vector_mask_path` is provided:
      - It is assumed to contain features with an 'image' field whose values match
        the image's filename (without extension).
      - Only pixels within those features (polygons) are considered valid; all
        others are set to the image's NoData value.

    Args:
        num_bands (int): Number of bands in the images.
        input_image_path_i (str): Path to the first image.
        input_image_path_j (str): Path to the second image.
        id_i (int): ID of the first image.
        id_j (int): ID of the second image.
        bound_i (dict): Bounds of the first image in the format
            {"x_min", "x_max", "y_min", "y_max"}.
        bound_j (dict): Bounds of the second image in the format
            {"x_min", "x_max", "y_min", "y_max"}.
        nodata_i (float): The NoData value for the first image.
        nodata_j (float): The NoData value for the second image.
        vector_mask_path (str, optional): File path to a GPKG containing mask
            polygons. Only polygons whose 'image' attribute matches the
            base name of the image file (excluding extension) are used
            to define valid pixels. Pixels outside these polygons are
            set to the image’s NoData value. If None, no masking is applied.

    Returns:
        tuple: A tuple containing:
            - overlap_stat: Dictionary of overlap statistics in the format:
              {
                id_i: {
                  id_j: {
                    band: {'mean': value, 'std': value, 'size': value}
                  }
                },
                id_j: {
                  id_i: {
                    band: {'mean': value, 'std': value, 'size': value}
                  }
                }
              }
            - whole_stats: Dictionary of whole image statistics in the format:
              {
                id_i: {
                  band: {'mean': value, 'std': value, 'size': value}
                },
                id_j: {
                  band: {'mean': value, 'std': value, 'size': value}
                }
              }
    """
    # Initialize the result dictionaries
    overlap_stat = {id_i: {id_j: {}}, id_j: {id_i: {}}}
    whole_stats = {id_i: {}, id_j: {}}

    # --- Helper function to rasterize a relevant mask from GPKG and apply it ---
    def _create_mask_from_gpkg(dataset, image_path):
        """
        If vector_mask_path is not None, rasterize the polygons from the GPKG where
        the 'image' field contains the base_name of the given image_path (split by ';'),
        and return a mask array (1 where valid). Pixels outside those polygons
        are set to NoData in the caller.

        Args:
            dataset (gdal.Dataset): The opened raster dataset (for size/projection).
            image_path (str): Path to the raster, used to derive the base_name.

        Returns:
            np.ndarray or None: A mask array (same shape as raster) with 1 for valid
                                pixels and 0 for invalid. If no matching features or
                                vector_mask_path is None, returns None.
        """
        if vector_mask_path is None:
            return None  # No masking needed

        # Extract base filename (no extension) to match the 'image' field
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        gpkg_ds = ogr.Open(vector_mask_path)
        if not gpkg_ds:
            raise RuntimeError(f"Could not open vector file: {vector_mask_path}")

        layer = gpkg_ds.GetLayer(0)  # Adjust if multiple layers or unknown index

        # We'll build an in-memory layer containing only the features
        # whose 'image' field includes base_name when split by ';'
        mem_driver_ogr = ogr.GetDriverByName("Memory")
        mem_ds = mem_driver_ogr.CreateDataSource("")
        mem_layer = mem_ds.CreateLayer("temp", layer.GetSpatialRef(), geom_type=ogr.wkbPolygon)

        # Replicate the field structure (not strictly required if only geometry is needed)
        layer_def = layer.GetLayerDefn()
        for i in range(layer_def.GetFieldCount()):
            mem_layer.CreateField(layer_def.GetFieldDefn(i))

        # Loop through features, split the 'image' field by ';',
        # and copy those whose list contains base_name
        layer.ResetReading()
        for feat in layer:
            images_field = feat.GetField("image")
            if images_field:
                image_list = [val.strip() for val in images_field.split(";")]
                if base_name in image_list:
                    # Create new feature in the memory layer
                    new_feat = ogr.Feature(mem_layer.GetLayerDefn())
                    new_feat.SetGeometry(feat.GetGeometryRef().Clone())
                    # Optionally copy over all field values
                    for fld_idx in range(layer_def.GetFieldCount()):
                        new_feat.SetField(fld_idx, feat.GetField(fld_idx))
                    mem_layer.CreateFeature(new_feat)
                    new_feat = None

        # If no features matched, return None
        if mem_layer.GetFeatureCount() == 0:
            return None

        # Prepare in-memory raster for the mask
        xsize = dataset.RasterXSize
        ysize = dataset.RasterYSize
        geo_transform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

        mem_driver_gdal = gdal.GetDriverByName("MEM")
        mask_ds = mem_driver_gdal.Create("", xsize, ysize, 1, gdal.GDT_Byte)
        mask_ds.SetGeoTransform(geo_transform)
        mask_ds.SetProjection(projection)

        # Initialize mask to 0
        mask_ds.GetRasterBand(1).Fill(0)

        # Rasterize polygons with a burn value of 1 for valid pixels
        err = gdal.RasterizeLayer(
            mask_ds,
            [1],
            mem_layer,
            burn_values=[1],
            options=["ALL_TOUCHED=FALSE"]  # or TRUE if needed
        )
        if err != gdal.CE_None:
            raise RuntimeError("Rasterize failed.")

        # Convert to NumPy array
        mask_array = mask_ds.GetRasterBand(1).ReadAsArray()
        return mask_array

    # Open both images
    dataset_i = gdal.Open(input_image_path_i)
    dataset_j = gdal.Open(input_image_path_j)

    if not dataset_i or not dataset_j:
        raise RuntimeError("Failed to open one or both input images.")

    geo_transform_i = dataset_i.GetGeoTransform()
    geo_transform_j = dataset_j.GetGeoTransform()

    # Pre-rasterize masks (if any) so we do it once per image, not per band
    mask_array_i = _create_mask_from_gpkg(dataset_i, input_image_path_i)
    if mask_array_i is not None: print(f"Found vector mask for {input_image_path_i}")

    mask_array_j = _create_mask_from_gpkg(dataset_j, input_image_path_j)
    if mask_array_j is not None: print(f"Found vector mask for {input_image_path_j}")

    # Calculate overlap bounds (do not change existing logic)
    x_min_overlap = max(bound_i["x_min"], bound_j["x_min"])
    x_max_overlap = min(bound_i["x_max"], bound_j["x_max"])
    y_min_overlap = max(bound_i["y_min"], bound_j["y_min"])
    y_max_overlap = min(bound_i["y_max"], bound_j["y_max"])

    if x_min_overlap >= x_max_overlap or y_min_overlap >= y_max_overlap:
        # Return empty stats if no overlap
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
        # Read entire band data
        band_data_i = dataset_i.GetRasterBand(band_idx + 1).ReadAsArray()
        band_data_j = dataset_j.GetRasterBand(band_idx + 1).ReadAsArray()

        # Calculate WHOLE IMAGE stats ignoring nodata only
        mask_i_no_mask = band_data_i != nodata_i
        valid_data_i_no_mask = band_data_i[mask_i_no_mask]
        whole_stats[id_i][band_idx] = {
            "mean": float(np.mean(valid_data_i_no_mask)) if valid_data_i_no_mask.size > 0 else 0.0,
            "std": float(np.std(valid_data_i_no_mask)) if valid_data_i_no_mask.size > 0 else 0.0,
            "size": int(valid_data_i_no_mask.size),
        }

        mask_j_no_mask = band_data_j != nodata_j
        valid_data_j_no_mask = band_data_j[mask_j_no_mask]
        whole_stats[id_j][band_idx] = {
            "mean": float(np.mean(valid_data_j_no_mask)) if valid_data_j_no_mask.size > 0 else 0.0,
            "std": float(np.std(valid_data_j_no_mask)) if valid_data_j_no_mask.size > 0 else 0.0,
            "size": int(valid_data_j_no_mask.size),
        }

        # If vector masks exist, apply them (set outside to NoData)
        if mask_array_i is not None:
            outside_mask = (mask_array_i == 0)
            band_data_i[outside_mask] = nodata_i

        if mask_array_j is not None:
            outside_mask = (mask_array_j == 0)
            band_data_j[outside_mask] = nodata_j

        # Compute overlap stats (now masked if mask_array is used)
        overlap_data_i = band_data_i[row_min_i:row_max_i, col_min_i:col_max_i]
        overlap_data_j = band_data_j[row_min_j:row_max_j, col_min_j:col_max_j]

        overlap_mask_i = overlap_data_i != nodata_i
        overlap_mask_j = overlap_data_j != nodata_j
        overlap_mask = overlap_mask_i & overlap_mask_j

        overlap_data_combined_i = overlap_data_i[overlap_mask]
        overlap_data_combined_j = overlap_data_j[overlap_mask]

        mean1 = np.mean(overlap_data_combined_i) if overlap_data_combined_i.size > 0 else 0
        std1 = np.std(overlap_data_combined_i) if overlap_data_combined_i.size > 0 else 0
        mean2 = np.mean(overlap_data_combined_j) if overlap_data_combined_j.size > 0 else 0
        std2 = np.std(overlap_data_combined_j) if overlap_data_combined_j.size > 0 else 0
        overlap_size = int(overlap_mask.sum())

        overlap_stat[id_i][id_j][band_idx] = {
            "mean": float(mean1),
            "std": float(std1),
            "size": overlap_size,
        }
        overlap_stat[id_j][id_i][band_idx] = {
            "mean": float(mean2),
            "std": float(std2),
            "size": overlap_size,
        }

    # Clean up
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


