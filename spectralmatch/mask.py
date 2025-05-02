import os
import rasterio
import geopandas as gpd
import numpy as np

from rasterio.enums import Resampling
from rasterio.transform import from_origin
from omnicloudmask import predict_from_array
from rasterio.features import shapes
from osgeo import gdal, ogr, osr

def create_cloud_mask_with_omnicloudmask(
    input_image_path,
    red_band_index,
    green_band_index,
    nir_band_index, # Blue band can work if nir isnt available
    output_mask_path,
    down_sample_m=None, # Down sample to 10 m if imagery has a spatial resolution < 10 m
    ):
    """
    Generates a cloud mask using OmniCloudMask from a multi-band image.

    Args:
    input_image_path (str): Path to the input image.
    red_band_index (int): Index of the red band.
    green_band_index (int): Index of the green band.
    nir_band_index (int): Index of the NIR (or substitute blue) band.
    output_mask_path (str): Path to save the output cloud mask GeoTIFF.
    down_sample_m (float, optional): Target resolution (in meters) to downsample the input before processing.

    Outputs:
    Saves a single-band cloud mask GeoTIFF at the specified path.
    """

    with rasterio.open(input_image_path) as src:
        if down_sample_m is not None:
            # Compute new dimensions based on the image bounds and the desired resolution.
            left, bottom, right, top = src.bounds
            new_width = int((right - left) / down_sample_m)
            new_height = int((top - bottom) / down_sample_m)
            new_transform = from_origin(left, top, down_sample_m, down_sample_m)
            # Read the bands with resampling to the new size.
            red   = src.read(red_band_index, out_shape=(new_height, new_width),
                             resampling=Resampling.bilinear)
            green = src.read(green_band_index, out_shape=(new_height, new_width),
                             resampling=Resampling.bilinear)
            nir   = src.read(nir_band_index, out_shape=(new_height, new_width),
                             resampling=Resampling.bilinear)
            meta = src.meta.copy()
            meta.update({
                'width': new_width,
                'height': new_height,
                'transform': new_transform,
            })
        else:
            # Read without resampling.
            red   = src.read(red_band_index)
            green = src.read(green_band_index)
            nir   = src.read(nir_band_index)
            meta = src.meta.copy()

        # Stack bands into an array of shape (3, height, width).
        band_array = np.stack([red, green, nir], axis=0)

    # Predict the mask (expected shape: (1, height, width))
    pred_mask = predict_from_array(band_array)
    pred_mask = np.squeeze(pred_mask)

    # Update metadata for a single-band output.
    meta.update({
        'driver': 'GTiff',
        'count': 1,
        'dtype': pred_mask.dtype,
        'nodata': 0,
    })

    # Write the predicted mask to a GeoTIFF file.
    with rasterio.open(output_mask_path, 'w', **meta) as dst:
        dst.write(pred_mask, 1)


def post_process_raster_cloud_mask_to_vector(
    input_image_path: str,
    minimum_mask_size_percentile: float = None,
    polygon_buffering_in_map_units: dict = None,
    value_mapping: dict = None
    ) -> ogr.DataSource:
    """
    Converts a raster cloud mask to a vector layer with optional filtering, buffering, and merging.

    Args:
    input_image_path (str): Path to the input cloud mask raster.
    minimum_mask_size_percentile (float, optional): Percentile threshold to filter small polygons by area.
    polygon_buffering_in_map_units (dict, optional): Mapping of raster values to buffer distances.
    value_mapping (dict, optional): Mapping of original raster values to new values before vectorization.

    Returns:
    ogr.DataSource: In-memory vector layer with merged and filtered polygons.

    Outputs:
    Returns an OGR DataSource containing post-processed vector features.
    """

    with rasterio.open(input_image_path) as src:
        raster_data = src.read(1)
        transform = src.transform
        crs = src.crs

    if value_mapping is not None:
        mapped = np.copy(raster_data)
        for orig_value, new_value in value_mapping.items():
            mapped[raster_data == orig_value] = new_value
        raster_data = mapped

    results = (
        {'properties': {'value': v}, 'geometry': s}
        for s, v in shapes(raster_data, transform=transform, connectivity=4)
    )
    features = list(results)
    if not features:
        print("No features were detected in the raster mask.")
        return None


    gdf = gpd.GeoDataFrame.from_features(features, crs=crs)

    gdf['area'] = gdf.geometry.area
    if minimum_mask_size_percentile is not None:
        area_threshold = np.percentile(gdf['area'], minimum_mask_size_percentile)
        print(f"Area threshold (at {minimum_mask_size_percentile}th percentile): {area_threshold:.2f}")
        gdf = gdf[gdf['area'] >= area_threshold].copy()

    if polygon_buffering_in_map_units is not None:
        gdf['geometry'] = gdf.apply(
            lambda row: row['geometry'].buffer(polygon_buffering_in_map_units.get(row['value'], 0))
            if row['value'] in polygon_buffering_in_map_units else row['geometry'],
            axis=1
        )

    merged_features = []
    for val, group in gdf.groupby('value'):
        # Use union_all() to merge the geometries within the group.
        # (Requires Shapely 2.0 or later; otherwise use shapely.ops.unary_union on group.geometry.tolist())
        union_geom = group.geometry.union_all()
        # If the union produces a single Polygon, add it directly;
        # if it produces a MultiPolygon, split it into individual features.
        if union_geom.geom_type == 'Polygon':
            merged_features.append({'value': val, 'geometry': union_geom})
        elif union_geom.geom_type == 'MultiPolygon':
            for geom in union_geom.geoms:
                merged_features.append({'value': val, 'geometry': geom})
        else:
            # In case of unexpected geometry types, skip or handle accordingly.
            print(f"Unexpected geometry type for value {val}: {union_geom.geom_type}")
    # Create a new GeoDataFrame from merged features.
    gdf = gpd.GeoDataFrame(merged_features, crs=gdf.crs)


    ogr_driver = ogr.GetDriverByName("Memory")
    mem_ds = ogr_driver.CreateDataSource("in_memory")

    # Determine an appropriate OGR geometry type using the first feature.
    first_geom = gdf.geometry.iloc[0]
    if first_geom.geom_type == "Polygon":
        ogr_geom_type = ogr.wkbPolygon
    elif first_geom.geom_type == "MultiPolygon":
        ogr_geom_type = ogr.wkbMultiPolygon
    else:
        ogr_geom_type = ogr.wkbUnknown

    # Convert the CRS to OGR SpatialReference.
    sr = osr.SpatialReference()
    try:
        sr.ImportFromWkt(crs.to_wkt())
    except AttributeError:
        sr.ImportFromEPSG(4326)

    mem_layer = mem_ds.CreateLayer("post_processed", sr, ogr_geom_type)

    # Add attribute field for 'value' (and any other non-geometry columns if needed).
    # Here we add 'value' for example.
    field_defn = ogr.FieldDefn("value", ogr.OFTInteger)
    mem_layer.CreateField(field_defn)

    # Add each row from the GeoDataFrame as an OGR feature.
    for idx, row in gdf.iterrows():
        feat = ogr.Feature(mem_layer.GetLayerDefn())
        ogr_geom = ogr.CreateGeometryFromWkt(row['geometry'].wkt)
        feat.SetGeometry(ogr_geom)
        feat.SetField("value", row['value'])
        mem_layer.CreateFeature(feat)
        feat = None

    return mem_ds


def create_ndvi_mask(
    input_image_path: str,
    output_image_path: str,
    nir_band: int=4,
    red_band: int=3,
    ):
    """
    Computes NDVI from a multi-band image and saves the result as a VRT raster.

    Args:
    input_image_path (str): Path to the input image with NIR and red bands.
    output_image_path (str): Path to save the NDVI output as a VRT file.
    nir_band (int, optional): Band index for NIR. Defaults to 4.
    red_band (int, optional): Band index for red. Defaults to 3.

    Returns:
    str: Path to the saved NDVI output.
    """

    ds = gdal.Open(input_image_path)
    nir = ds.GetRasterBand(nir_band).ReadAsArray().astype(np.float32)
    red = ds.GetRasterBand(red_band).ReadAsArray().astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-9)  # avoid division by zero

    mem_drv = gdal.GetDriverByName("MEM")
    mem_ds = mem_drv.Create("", ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32)
    mem_ds.SetGeoTransform(ds.GetGeoTransform())
    mem_ds.SetProjection(ds.GetProjection())
    mem_ds.GetRasterBand(1).WriteArray(ndvi)

    gdal.GetDriverByName("VRT").CreateCopy(output_image_path, mem_ds)
    ds, mem_ds = None, None
    return output_image_path


def post_process_threshold_to_vector(
    input_image_path: str,
    output_vector_path: str,
    threshold_val: float | int,
    operator_str: str="<=",
    ):
    """
    Converts a thresholded raster mask to a vector layer based on a comparison operator.

    Args:
    input_image_path (str): Path to the input single-band raster.
    output_vector_path (str): Path to save the output vector file (GeoPackage).
    threshold_val (float | int): Threshold value to apply.
    operator_str (str, optional): Comparison operator ('<=', '>=', '<', '>', '=='). Defaults to '<='.

    Returns:
    str: Path to the saved vector file.

    Raises:
    ValueError: If an unsupported comparison operator is provided.
    """

    ds = gdal.Open(input_image_path)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()

    if operator_str == "<=":
        mask = arr <= threshold_val
    elif operator_str == ">=":
        mask = arr >= threshold_val
    elif operator_str == "<":
        mask = arr < threshold_val
    elif operator_str == ">":
        mask = arr > threshold_val
    elif operator_str == "==":
        mask = arr == threshold_val
    else:
        raise ValueError("Unsupported operator")

    mask = mask.astype(np.uint8)

    mem_ds = gdal.GetDriverByName("MEM").Create("", ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)
    mem_ds.SetGeoTransform(ds.GetGeoTransform())
    mem_ds.SetProjection(ds.GetProjection())
    mem_ds.GetRasterBand(1).WriteArray(mask)

    drv = ogr.GetDriverByName("GPKG")
    if os.path.exists(output_vector_path):
        drv.DeleteDataSource(output_vector_path)
    out_ds = drv.CreateDataSource(output_vector_path)
    out_lyr = out_ds.CreateLayer("mask", srs=None)
    out_lyr.CreateField(ogr.FieldDefn("DN", ogr.OFTInteger))

    gdal.Polygonize(mem_ds.GetRasterBand(1), mem_ds.GetRasterBand(1), out_lyr, 0, [])
    ds, mem_ds, out_ds = None, None, None
    return output_vector_path