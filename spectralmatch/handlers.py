import os
import tempfile
import rasterio
import numpy as np
import tempfile
import os
import rasterio

from typing import Tuple, Optional
from osgeo import ogr
from rasterio.windows import Window
from rasterio.transform import from_bounds
from rasterio.warp import aligned_target
from rasterio.warp import reproject
from rasterio.enums import Resampling
from typing import Literal
from spectralmatch.utils.utils_common import _create_windows

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

def merge_rasters(
    data_in,
    data_out: str,
    resampling_method: Literal["nearest", "bilinear", "cubic"] = "nearest",
    tap: bool = False,
    resolution: Literal["highest", "lowest"] = "highest",
    tile_width_and_height_tuple: Optional[Tuple[int, int]] = None
    ):

    srcs = [rasterio.open(path) for path in data_in]
    resampling_enum = {"nearest": Resampling.nearest, "bilinear": Resampling.bilinear, "cubic": Resampling.cubic}[resampling_method]

    # Determine target resolution
    all_resolutions = [src.res for src in srcs]
    if resolution == "highest":
        target_res = min(all_resolutions, key=lambda r: r[0] * r[1])
    elif resolution == "lowest":
        target_res = max(all_resolutions, key=lambda r: r[0] * r[1])
    else:
        raise ValueError("resolution must be 'highest' or 'lowest'")

    pixel_width, pixel_height = target_res

    # Compute bounds of the output mosaic
    lefts, bottoms, rights, tops = zip(*[src.bounds for src in srcs])
    minx, miny, maxx, maxy = min(lefts), min(bottoms), max(rights), max(tops)

    if tap:
        minx = np.floor(minx / pixel_width) * pixel_width
        maxx = np.ceil(maxx / pixel_width) * pixel_width
        miny = np.floor(miny / abs(pixel_height)) * abs(pixel_height)
        maxy = np.ceil(maxy / abs(pixel_height)) * abs(pixel_height)

    width = int(np.round((maxx - minx) / pixel_width))
    height = int(np.round((maxy - miny) / abs(pixel_height)))
    out_transform = from_bounds(minx, miny, maxx, maxy, width, height)

    out_crs = srcs[0].crs
    nodata = srcs[0].nodata or 0
    num_bands = srcs[0].count
    dtype = srcs[0].dtypes[0]

    out_meta = {
        "driver": "GTiff",
        "count": num_bands,
        "dtype": dtype,
        "nodata": nodata,
        "width": width,
        "height": height,
        "transform": out_transform,
        "crs": out_crs
    }

    with rasterio.open(data_out, "w", **out_meta) as data_out:
        if tile_width_and_height_tuple:
            windows = _create_windows(width, height, *tile_width_and_height_tuple)
        else:
            windows = [Window(0, 0, width, height)]

        for window in windows:
            dst_array = np.full((num_bands, int(window.height), int(window.width)), nodata, dtype=dtype)

            for src in srcs:
                for b in range(1, num_bands + 1):
                    temp = np.full((int(window.height), int(window.width)), nodata, dtype=dtype)
                    reproject(
                        source=rasterio.band(src, b),
                        destination=temp,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=rasterio.windows.transform(window, out_transform),
                        dst_crs=out_crs,
                        dst_nodata=nodata,
                        resampling=resampling_enum
                    )
                    # Only replace where temp is not nodata
                    mask = temp != nodata
                    dst_array[b - 1][mask] = temp[mask]

            data_out.write(dst_array, window=window)

    print(f"Merged raster saved to: {data_out}")

def align_rasters(
    input_image_paths: list[str],
    resample_method: Literal["nearest", "bilinear", "cubic", "average", "mode", "max", "min", "med", "q1", "q3"] = "bilinear",
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