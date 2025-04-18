
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from omnicloudmask import predict_from_array
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from osgeo import ogr, osr
from spectralmatch.handlers import write_vector
import os
from osgeo import gdal, ogr
import numpy as np

def create_cloud_mask(
    input_image_path,
    red_band_index,
    green_band_index,
    nir_band_index, # Blue band can work if nir isnt available
    output_mask_path,
    down_sample_m=None, # Down sample to 10 m if imagery has a spatial resolution < 10 m
    ):
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

def post_process_raster_cloud_mask(
        input_image: str,
        minimum_mask_size_percentile: float = None,
        polygon_buffering_in_map_units: dict = None,
        value_mapping: dict = None
) -> ogr.DataSource:
    """
Vectorizes a cloud mask raster and post-processes the polygons.

Parameters:
input_image (str): Path to the input mask raster (e.g., a TIFF).
minimum_mask_size_percentile (float, optional): Percentile threshold; polygons whose area is below
this percentile are removed. If None, no area-based filtering is applied.
polygon_buffering_in_map_units (dict, optional): A dictionary mapping a polygon's 'value' attribute
(from vectorization) to a buffering distance in map units.
For example: {0: 5, 1: 30} buffers polygons with value 0 by 5 units and those with value 1 by 30 units.
If a polygon's value is not in the dictionary, its geometry remains unchanged.
value_mapping (dict, optional): A dictionary mapping original pixel values to new group values.
For example: {1: 1, 2: 1, 3: 1} clusters pixels with values 1, 2, and 3 together.

Returns:
ogr.DataSource: An in-memory GDAL vector datasource containing the processed polygon features.

The function performs:
1. Reading the first band of the raster mask.
2. Optional value mapping.
3. Vectorization of the raster into polygons.
4. (Optional) Removal of polygons with areas below the given percentile.
5. Buffering for each polygon based on the provided dictionary.
6. Merging of overlapping polygons with the same "value".
7. Conversion into an in-memory GDAL vector datasource.
    """
    # --- Step 1: Read the raster mask ---
    with rasterio.open(input_image) as src:
        raster_data = src.read(1)
        transform = src.transform
        crs = src.crs  # Rasterio CRS (usually a pyproj CRS)

    # --- Step 2: Apply optional value mapping ---
    if value_mapping is not None:
        mapped = np.copy(raster_data)
        for orig_value, new_value in value_mapping.items():
            mapped[raster_data == orig_value] = new_value
        raster_data = mapped

    # --- Step 3: Vectorize the raster ---
    results = (
        {'properties': {'value': v}, 'geometry': s}
        for s, v in shapes(raster_data, transform=transform, connectivity=4)
    )
    features = list(results)
    if not features:
        print("No features were detected in the raster mask.")
        return None

    # Create a GeoDataFrame from the vectorized features.
    gdf = gpd.GeoDataFrame.from_features(features, crs=crs)

    # --- Step 4: Compute areas and, if requested, filter out small polygons ---
    gdf['area'] = gdf.geometry.area
    if minimum_mask_size_percentile is not None:
        area_threshold = np.percentile(gdf['area'], minimum_mask_size_percentile)
        print(f"Area threshold (at {minimum_mask_size_percentile}th percentile): {area_threshold:.2f}")
        gdf = gdf[gdf['area'] >= area_threshold].copy()

    # --- Step 5: Apply buffering per polygon based on the provided dictionary ---
    if polygon_buffering_in_map_units is not None:
        gdf['geometry'] = gdf.apply(
            lambda row: row['geometry'].buffer(polygon_buffering_in_map_units.get(row['value'], 0))
            if row['value'] in polygon_buffering_in_map_units else row['geometry'],
            axis=1
        )

    # --- Step 6: Merge overlapping polygons by 'value' ---
    # Group by the 'value' attribute and merge (union) polygons within each group.
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

    # --- Step 7: Convert the GeoDataFrame to an in-memory GDAL vector datasource ---
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
    # Optionally, add other fields (e.g. 'area') if desired.

    # Add each row from the GeoDataFrame as an OGR feature.
    for idx, row in gdf.iterrows():
        feat = ogr.Feature(mem_layer.GetLayerDefn())
        ogr_geom = ogr.CreateGeometryFromWkt(row['geometry'].wkt)
        feat.SetGeometry(ogr_geom)
        feat.SetField("value", row['value'])
        mem_layer.CreateFeature(feat)
        feat = None

    return mem_ds

def create_ndvi_vrt(input_image, nir_band=4, red_band=3, output_vrt="ndvi.vrt"):
    ds = gdal.Open(input_image)
    nir = ds.GetRasterBand(nir_band).ReadAsArray().astype(np.float32)
    red = ds.GetRasterBand(red_band).ReadAsArray().astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-9)  # avoid division by zero

    mem_drv = gdal.GetDriverByName("MEM")
    mem_ds = mem_drv.Create("", ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32)
    mem_ds.SetGeoTransform(ds.GetGeoTransform())
    mem_ds.SetProjection(ds.GetProjection())
    mem_ds.GetRasterBand(1).WriteArray(ndvi)

    gdal.GetDriverByName("VRT").CreateCopy(output_vrt, mem_ds)
    ds, mem_ds = None, None
    return output_vrt

def create_threshold_vector(ndvi_vrt, threshold_val, operator_str="<=", output_gpkg="mask.gpkg"):
    ds = gdal.Open(ndvi_vrt)
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
    if os.path.exists(output_gpkg):
        drv.DeleteDataSource(output_gpkg)
    out_ds = drv.CreateDataSource(output_gpkg)
    out_lyr = out_ds.CreateLayer("mask", srs=None)
    out_lyr.CreateField(ogr.FieldDefn("DN", ogr.OFTInteger))

    gdal.Polygonize(mem_ds.GetRasterBand(1), mem_ds.GetRasterBand(1), out_lyr, 0, [])
    ds, mem_ds, out_ds = None, None, None
    return output_gpkg

if __name__ == "__main__":
    create_cloud_mask(
        "/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/20171019_40cm_WV03_BAB_050311752010/Mul_OrthoFromUSGSLidar/17OCT19211817-M1BS-050311752010_01_P001_OrthoFromUSGSLidar.tif",
        5,
        3,
        8,
        "/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/20171019_40cm_WV03_BAB_050311752010/Masks/17OCT19211817-M1BS-050311752010_01_P001_OrthoFromUSGSLidar_CloudMask.tif",
        down_sample_m=10
    )
    write_vector(
            post_process_raster_cloud_mask(
                "/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/20171019_40cm_WV03_BAB_050311752010/Masks/17OCT19211817-M1BS-050311752010_01_P001_OrthoFromUSGSLidar_CloudMask.tif",
                None,
                {1: 50},
                {0: 0, 1: 1, 2: 1, 3: 1}
        ),
        "/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/20171019_40cm_WV03_BAB_050311752010/Masks/17OCT19211817-M1BS-050311752010_01_P001_OrthoFromUSGSLidar_CloudMask.gpkg",
    )

    input_tif = "input.tif"
    nir_band_num = 4
    red_band_num = 3
    ndvi_vrt = "ndvi_output.vrt"
    threshold_value = 0.2
    operator_text = "<="
    out_gpkg = "low_veg_mask.gpkg"

    create_ndvi_vrt(input_tif, nir_band_num, red_band_num, ndvi_vrt)
    create_threshold_vector(ndvi_vrt, threshold_value, operator_text, out_gpkg)