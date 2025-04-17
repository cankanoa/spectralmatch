import rasterio
from osgeo import ogr
from rasterio.windows import Window
import numpy as np
from rasterio.windows import Window
import os
import shutil
import tempfile
from typing import Literal
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, aligned_target

def create_windows(width, height, tile_width, tile_height):
    for row_off in range(0, height, tile_height):
        for col_off in range(0, width, tile_width):
            win_width = min(tile_width, width - col_off)
            win_height = min(tile_height, height - row_off)
            yield Window(col_off, row_off, win_width, win_height)

def _align_rasters(
    input_image_paths: list[str],
    resample_method: Literal[
        "nearest", "bilinear", "cubic", "average", "mode", "max", "min", "med", "q1", "q3"
    ] = "bilinear",
    tap: bool = True,
) -> list[str]:
    temp_dir = tempfile.mkdtemp()  # Persistent temp directory
    aligned_paths = []

    # 1. Determine highest resolution
    best_resolution = float("inf")
    for path in input_image_paths:
        with rasterio.open(path) as src:
            res = min(abs(src.transform.a), abs(src.transform.e))
            if res < best_resolution:
                best_resolution = res
    target_res = (best_resolution, best_resolution)

    # 2. Reproject each image to its own aligned grid using target resolution
    for path in input_image_paths:
        filename = os.path.basename(path)
        output_path = os.path.join(temp_dir, filename)

        with rasterio.open(path) as src:
            dst_crs = src.crs
            transform = src.transform
            width = src.width
            height = src.height

            if tap:
                dst_transform, dst_width, dst_height = aligned_target(
                    transform, width, height, target_res
                )
            else:
                dst_transform, dst_width, dst_height = transform, width, height

            profile = src.profile.copy()
            profile.update({
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
                "crs": dst_crs,
            })

            with rasterio.open(output_path, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=getattr(Resampling, resample_method),
                    )

        aligned_paths.append(output_path)

    return aligned_paths

def write_vector(
        mem_ds: ogr.DataSource,
        output_vector_path: str
) -> None:
    """
Writes an in-memory vector datasource to disk.
The driver is chosen based on the file extension of output_vector_path.
All layers, including metadata, schema, and features, are preserved.
    """
    # Map file extensions to OGR driver names for common vector formats.
    driver_mapping = {
        '.shp': 'ESRI Shapefile',
        '.geojson': 'GeoJSON',
        '.gpkg': 'GPKG'
    }
    ext = os.path.splitext(output_vector_path)[1].lower()
    driver_name = driver_mapping.get(ext, 'GeoJSON')  # Fallback to GeoJSON if unknown.

    driver = ogr.GetDriverByName(driver_name)
    if driver is None:
        raise RuntimeError(f"No driver found for extension: {ext}")

    # If the output file already exists, delete it.
    if os.path.exists(output_vector_path):
        driver.DeleteDataSource(output_vector_path)

    out_ds = driver.CreateDataSource(output_vector_path)
    if out_ds is None:
        raise RuntimeError(f"Could not create output vector dataset: {output_vector_path}")

    # Loop over every layer in the in-memory datasource and copy it.
    for i in range(mem_ds.GetLayerCount()):
        mem_layer = mem_ds.GetLayerByIndex(i)
        layer_name = mem_layer.GetName()
        srs = mem_layer.GetSpatialRef()
        geom_type = mem_layer.GetGeomType()

        out_layer = out_ds.CreateLayer(layer_name, srs, geom_type)

        # Copy field definitions
        mem_defn = mem_layer.GetLayerDefn()
        for j in range(mem_defn.GetFieldCount()):
            field_defn = mem_defn.GetFieldDefn(j)
            out_layer.CreateField(field_defn)

        # Copy features (including geometry, fields, and feature-level metadata)
        mem_layer.ResetReading()
        for feat in mem_layer:
            out_feat = ogr.Feature(out_layer.GetLayerDefn())
            out_feat.SetGeometry(feat.GetGeometryRef().Clone())
            for j in range(mem_defn.GetFieldCount()):
                field_name = mem_defn.GetFieldDefn(j).GetNameRef()
                out_feat.SetField(field_name, feat.GetField(j))
            out_layer.CreateFeature(out_feat)
            out_feat = None
    out_ds.Destroy()