import os
import rasterio
import numpy as np
import fiona
import geopandas as gpd
import re

from rasterio.enums import Resampling
from rasterio.transform import from_origin
from omnicloudmask import predict_from_array
from osgeo import gdal, ogr, osr
from shapely.geometry import shape, Polygon, MultiPolygon, mapping
from typing import Literal, Any, Tuple
from rasterio.features import shapes
from concurrent.futures import as_completed

from ..utils_multiprocessing import _get_executor, WorkerContext, _resolve_windows, _resolve_parallel_config
from ..handlers import _resolve_paths, _resolve_nodata_value, _resolve_output_dtype
from ..types_and_validation import Universal


def band_math(
    input_images: Universal.SearchFolderOrListFiles,
    output_images: Universal.CreateInFolderOrListFiles,
    custom_math: str,
    *,
    debug_logs: Universal.DebugLogs = False,
    custom_nodata_value: Universal.CustomNodataValue = None,
    image_parallel_workers: Universal.ImageParallelWorkers = None,
    window_parallel_workers: Universal.WindowParallelWorkers = None,
    window_size: Universal.WindowSize = None,
    custom_output_dtype: Universal.CustomOutputDtype = None,
    calculation_dtype: Universal.CalculationDtype = None,
):
    input_image_paths = _resolve_paths("search", input_images)
    output_image_paths = _resolve_paths("create", output_images, (input_image_paths,))
    image_names = _resolve_paths("name", input_image_paths)

    with rasterio.open(input_image_paths[0]) as ds:
        nodata_value = _resolve_nodata_value(ds, custom_nodata_value)
        output_dtype = _resolve_output_dtype(ds, custom_output_dtype)

    image_parallel, image_backend, image_max_workers = _resolve_parallel_config(image_parallel_workers)

    # Extract referenced bands from custom_math (e.g., b1, b2, ...)
    band_indices = sorted({int(match[1:]) for match in re.findall(r"\bb\d+\b", custom_math)})

    image_args = [
        (in_path, out_path, name, custom_math, debug_logs, nodata_value, window_parallel_workers, window_size, band_indices, output_dtype, calculation_dtype)
        for in_path, out_path, name in zip(input_image_paths, output_image_paths, image_names)
    ]

    if image_parallel:
        with _get_executor(image_backend, image_max_workers) as executor:
            futures = [executor.submit(band_math_process_image, *arg) for arg in image_args]
            for future in as_completed(futures):
                future.result()
    else:
        for arg in image_args:
            band_math_process_image(*arg)


def band_math_process_image(
    input_image_path: str,
    output_image_path: str,
    name: str,
    custom_math: str,
    debug_logs: bool,
    nodata_value,
    window_parallel_workers,
    window_size,
    band_indices,
    output_dtype,
    calculation_dtype,
):
    with rasterio.open(input_image_path) as src:
        profile = src.profile.copy()
        profile.update(dtype=output_dtype, count=1)

        window_parallel, window_backend, window_max_workers = _resolve_parallel_config(window_parallel_workers)

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        with rasterio.open(output_image_path, "w", **profile) as dst:
            windows = _resolve_windows(src, window_size)
            args = [
                (name, window, custom_math, debug_logs, nodata_value, band_indices, calculation_dtype)
                for window in windows
            ]

            if window_parallel:
                with _get_executor(window_backend, window_max_workers, initializer=WorkerContext.init, initargs=({name: ("raster", input_image_path)},)) as executor:
                    futures = [executor.submit(band_math_process_window, *arg) for arg in args]
                    for future in futures:
                        band, window, data = future.result()
                        dst.write(data.astype(output_dtype), band, window=window)
            else:
                WorkerContext.init({name: ("raster", input_image_path)})
                for arg in args:
                    band, window, data = band_math_process_window(*arg)
                    dst.write(data.astype(output_dtype), band, window=window)
                WorkerContext.close()


def band_math_process_window(
    name: str,
    window: rasterio.windows.Window,
    custom_math: str,
    debug_logs: bool,
    nodata_value,
    band_indices,
    calculation_dtype
):
    ds = WorkerContext.get(name)

    bands = [ds.read(i, window=window).astype(calculation_dtype) for i in band_indices]
    band_vars = {f"b{i}": b for i, b in zip(band_indices, bands)}

    try:
        result = eval(custom_math, {"np": np}, band_vars).astype(calculation_dtype)
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression '{custom_math}': {e}")

    if nodata_value is not None:
        nodata_mask = np.any([b == nodata_value for b in bands], axis=0)
        result[nodata_mask] = np.nan

    return 1, window, result


def create_cloud_mask_with_omnicloudmask(
    input_image_path,
    red_band_index,
    green_band_index,
    nir_band_index, # Blue band can work if nir isnt available
    output_mask_path,
    down_sample_m=None, # Down sample to 10 m if imagery has a spatial resolution < 10 m
    debug_logs: bool = False,
    **omnicloud_kwargs: Any,
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
        debug_logs (bool, optional): Debug logs to console.
        omnicloud_kwargs: Forwards key word args to OmniCloudMask predict_from_array() function. Repo here: https://github.com/DPIRD-DMA/OmniCloudMask.

    Outputs:
        Saves a single-band cloud mask GeoTIFF at the specified path.
    """

    print("Start create omnicloudmask")
    if not os.path.exists(os.path.dirname(output_mask_path)): os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
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
    pred_mask = predict_from_array(band_array, **omnicloud_kwargs)
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
    output_vector_path: str,
    minimum_mask_size_percentile: float = None,
    polygon_buffering_in_map_units: dict = None,
    value_mapping: dict = None
    ) -> ogr.DataSource:
    """
    Converts a raster cloud mask to a vector layer with optional filtering, buffering, and merging.

    Args:
        input_image_path (str): Path to the input cloud mask raster.
        output_vector_path (str): Path to the output vector layer.
        minimum_mask_size_percentile (float, optional): Percentile threshold to filter small polygons by area.
        polygon_buffering_in_map_units (dict, optional): Mapping of raster values to buffer distances.
        value_mapping (dict, optional): Mapping of original raster values to new values before vectorization.

    Outputs:
        Saves a vector layer to the output path.
    """

    print("Start post-processing raster cloud mask")
    with rasterio.open(input_image_path) as src:
        raster_data = src.read(1)
        transform = src.transform
        crs = src.crs

    if value_mapping is not None:
        include_mask = np.full(raster_data.shape, True, dtype=bool)
        mapped = np.copy(raster_data)
        for orig_value, new_value in value_mapping.items():
            if new_value is None:
                include_mask &= raster_data != orig_value  # Exclude from processing
            else:
                mapped[raster_data == orig_value] = new_value
        raster_data = mapped
    else:
        include_mask = None

    results = (
        {'properties': {'value': v}, 'geometry': s}
        for s, v in shapes(raster_data, mask=include_mask, transform=transform, connectivity=4)
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

    driver = ogr.GetDriverByName("GPKG")
    if os.path.exists(output_vector_path):
        driver.DeleteDataSource(output_vector_path)
    out_ds = driver.CreateDataSource(output_vector_path)
    out_ds.CopyLayer(mem_layer, "post_processed")
    out_ds = None

    return output_vector_path


def create_ndvi_mask(
    input_image_path: str,
    output_image_path: str,
    nir_band: int,
    red_band: int,
    ) -> str:
    """
    Computes NDVI from a multi-band image and saves the result as a GeoTIFF.

    Args:
        input_image_path (str): Path to the input image with NIR and red bands.
        output_image_path (str): Path to save the NDVI output GeoTIFF.
        nir_band (int): Band index for NIR (1-based).
        red_band (int): Band index for red (1-based).

    Returns:
        str: Path to the saved NDVI output.
    """

    print("Start ndvi computation")
    if not os.path.exists(os.path.dirname(output_image_path)): os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    with rasterio.open(input_image_path) as src:
        nir = src.read(nir_band).astype(np.float32)
        red = src.read(red_band).astype(np.float32)
        ndvi = (nir - red) / (nir + red + 1e-9)

        print("NIR min/max:", np.nanmin(nir), np.nanmax(nir))
        print("Red min/max:", np.nanmin(red), np.nanmax(red))
        print("NDVI min/max:", np.nanmin(ndvi), np.nanmax(ndvi))

        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1)

        with rasterio.open(output_image_path, 'w', **profile) as dst:
            dst.write(ndvi, 1)

    return output_image_path


def post_process_threshold_to_vector(
    input_image_path: str,
    output_vector_path: str,
    threshold_val: float | int,
    operator_str: Literal["=", "<=", ">", ">=", "=="] = "<=",
    ) -> str:
    """
    Converts a thresholded raster mask to a vector layer using Rasterio and Fiona.

    Args:
        input_image_path (str): Path to the input single-band raster.
        output_vector_path (str): Path to save the output vector file (GeoPackage).
        threshold_val (float | int): Threshold value to apply.
        operator_str (str): One of the comparison operators.

    Returns:
        str: Path to the saved vector file.
    """
    print("Start post process threshold")

    with rasterio.open(input_image_path) as src:
        image = src.read(1)
        transform = src.transform
        crs = src.crs

        # Apply threshold
        if operator_str == "<":
            mask = image < threshold_val
        elif operator_str == "<=":
            mask = image <= threshold_val
        elif operator_str == ">":
            mask = image > threshold_val
        elif operator_str == ">=":
            mask = image >= threshold_val
        elif operator_str == "==":
            mask = image == threshold_val
        else:
            raise ValueError("Unsupported operator_str")

        mask = mask.astype(np.uint8)

        # Generate vector shapes
        results = []
        for s, v in shapes(mask, mask=mask, transform=transform):
            if v != 1:
                continue
            geom = shape(s)
            if isinstance(geom, Polygon):
                results.append({"properties": {"DN": int(v)}, "geometry": mapping(geom)})
            elif isinstance(geom, MultiPolygon):
                for part in geom.geoms:
                    results.append({"properties": {"DN": int(v)}, "geometry": mapping(part)})

        schema = {
            "geometry": "Polygon",
            "properties": {"DN": "int"},
        }

        if os.path.exists(output_vector_path):
            os.remove(output_vector_path)

        with fiona.open(
            output_vector_path, "w",
            driver="GPKG",
            crs=crs,
            schema=schema,
            layer="mask"
        ) as dst:
            for feat in results:
                dst.write(feat)

    return output_vector_path