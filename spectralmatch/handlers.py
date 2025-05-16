import os
import numpy as np
import tempfile
import rasterio
import shutil
import geopandas as gpd

from typing import Tuple, Optional, Literal, List, Optional, Literal, Tuple
from osgeo import ogr
from rasterio.windows import Window
from rasterio.transform import from_bounds
from rasterio.warp import aligned_target, reproject, transform_bounds
from rasterio.enums import Resampling
from .utils import _create_windows
from rasterio.mask import mask
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.coords import BoundingBox


def _resolve_input_output_paths(
    input_images_item: str | List[str],
    output_images_item: Tuple[str, str] | List[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Resolves input and output image paths to dictionaries keyed by image basename.

    Args:
        input_images_item (str | List[str]): Input folder to loop through or list of input image paths.
        output_images_item (Tuple[str, str] | List[str]): Either a tuple of (output folder, suffix) or a list of output paths.

    Returns:
        Tuple[Dict[str, str], Dict[str, str]]: (input_images, output_images)

    Raises:
        ValueError: If the number of input and output images does not match.
    """
    if isinstance(input_images_item, str):
        input_paths = [
            os.path.join(input_images_item, f)
            for f in os.listdir(input_images_item)
            if f.lower().endswith(".tif")
        ]
    else:
        input_paths = input_images_item

    input_images = {
        os.path.splitext(os.path.basename(p))[0]: p for p in input_paths
    }

    if isinstance(output_images_item, tuple):
        folder, suffix = output_images_item
        output_images = {
            name: os.path.join(folder, f"{name}{suffix}.tif")
            for name in input_images
        }
    else:
        output_images = {
            os.path.splitext(os.path.basename(p))[0]: p for p in output_images_item
        }

    if len(input_images) != len(output_images):
        raise ValueError(f"Input and output image counts do not match "
                         f"({len(input_images)} vs {len(output_images)}).")

    return input_images, output_images


def _write_vector(
    mem_ds: ogr.Layer,
    output_vector_path: str
    ) -> None:
    """
    Writes an in-memory OGR DataSource to disk in a supported vector format.

    Args:
        mem_ds (ogr.DataSource): In-memory vector data source.
        output_vector_path (str): Output file path (.shp, .geojson, or .gpkg).

    Returns:
        None

    Raises:
        RuntimeError: If no suitable driver is found or output creation fails.
    """

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
    input_image_paths: List[str],
    output_image_path: str,
    window_size: Optional[int | Tuple[int, int]] = None,
    debug_logs: bool = False,
    output_dtype: str | None = None,
    ) -> None:
    """
    Merges multiple input rasters into a single mosaic file by aligning each image geospatially and writing them in the correct location using tiling.

    Args:
        input_image_paths (List[str]): Paths to input raster images.
        output_image_path (str): Path to save the merged output raster.
        window_size (int | Tuple[int, int] | None, optional): Tile size for memory-efficient processing.
        debug_logs (bool, optional): Enable debug logging.
        output_dtype (str | None, optional): Output dtype for output raster. None will default to input raster type.

    Output:
        A geospatially aligned, merged raster is saved to `output_image_path`.
    """
    if debug_logs: print('Start merging')
    if not os.path.exists(os.path.dirname(output_image_path)):
        os.makedirs(os.path.dirname(output_image_path))

    if isinstance(window_size, int):
        window_size = (window_size, window_size)

    # Read metadata and calculate combined bounds and resolution
    all_bounds = []
    all_res = []
    crs = None
    dtype = None
    count = None

    for path in input_image_paths:
        with rasterio.open(path) as src:
            nodata_value = src.nodata

    for path in input_image_paths:
        with rasterio.open(path) as src:
            all_bounds.append(src.bounds)
            all_res.append(src.res)
            if crs is None:
                crs = src.crs
                dtype = output_dtype or src.dtypes[0]
                count = src.count

    minx = min(b.left for b in all_bounds)
    miny = min(b.bottom for b in all_bounds)
    maxx = max(b.right for b in all_bounds)
    maxy = max(b.top for b in all_bounds)

    res_x, res_y = all_res[0]  # Assume same resolution across rasters
    width = int(np.ceil((maxx - minx) / res_x))
    height = int(np.ceil((maxy - miny) / res_y))
    transform = from_origin(minx, maxy, res_x, res_y)

    if window_size:
        windows = list(_create_windows(width, height, *window_size))
    else:
        windows = [Window(0, 0, width, height)]

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata_value or None
    }
    # if debug_logs: print(f"xStart:yStart xSizeXySize ({len(windows)} windows): ", end="")
    with rasterio.open(output_image_path, 'w', **profile) as dst:
        for window in windows:
            win_transform = rasterio.windows.transform(window, transform)
            win_bounds = BoundingBox(*rasterio.windows.bounds(window, transform))
            merged_data = np.zeros((count, window.height, window.width), dtype=dtype)

            for path in input_image_paths:
                with rasterio.open(path) as src:
                    src_bounds = src.bounds
                    if (
                            src_bounds.right <= win_bounds.left or
                            src_bounds.left >= win_bounds.right or
                            src_bounds.top <= win_bounds.bottom or
                            src_bounds.bottom >= win_bounds.top
                    ):
                        continue
                    try:
                        src_window = rasterio.windows.from_bounds(
                            *win_bounds,
                            transform=src.transform
                        )
                        src_data = src.read(
                            window=src_window,
                            out_shape=(count, window.height, window.width),
                            resampling=Resampling.nearest
                        )

                        nodata_val = src.nodata
                        if nodata_val is not None:
                            mask = ~(np.isclose(src_data, nodata_val))
                        else:
                            mask = (src_data != 0)

                        merged_data = np.where(mask, src_data, merged_data)

                    except Exception as e:
                        if debug_logs:
                            print(f"Skipping {path} in window {window} due to error: {e}")

            dst.write(merged_data, window=window)
            # if debug_logs: print(f"{window.col_off}:{window.row_off} {window.width}x{window.height}, ", end="", flush=True)
    if debug_logs:
        print()
        print("Done merging")


def mask_rasters(
    input_image_paths: List[str],
    output_image_paths: List[str],
    vector_mask_path: str,
    split_mask_by_attribute: Optional[str] = None,
    resampling_method: Literal["nearest", "bilinear", "cubic"] = "nearest",
    tap: bool = False,
    resolution: Literal["highest", "average", "lowest"] = "highest",
    window_size: Optional[int | Tuple[int, int]] = None,
    debug_logs: bool = False,
    include_touched_pixels: bool = False,
    ) -> None:
    """
    Masks rasters using vector geometries. If `split_mask_by_attribute` is set,
    geometries are filtered by the raster's basename (excluding extension) to allow
    per-image masking with specific matching features.

    Args:
        input_image_paths (List[str]): Paths to input rasters.
        output_image_paths (List[str]): Corresponding output raster paths.
        vector_mask_path (str): Path to vector mask file (.shp, .gpkg, etc.).
        split_mask_by_attribute (Optional[str]): Attribute to match raster basenames.
        resampling_method (Literal["nearest", "bilinear", "cubic"]): Resampling algorithm.
        tap (bool): Snap output bounds to target-aligned pixels.
        resolution (Literal["highest", "average", "lowest"]): Strategy to determine target resolution.
        window_size (Optional[int | Tuple[int, int]]): Optional tile size for processing.
        debug_logs (bool): Print debug information if True.
        include_touched_pixels (bool): If True, include touched pixels in output raster.

    Outputs:
        Saved masked raster files to output_image_paths.
    """
    if debug_logs: print(f'Masking {len(input_image_paths)} rasters')

    gdf = gpd.read_file(vector_mask_path)

    if isinstance(window_size, int): window_size = (window_size, window_size)

    # Compute target resolution and bounds
    resolutions = []
    bounds_list = []
    crs_set = set()

    for path in input_image_paths:
        with rasterio.open(path) as src:
            res_x, res_y = src.res
            resolutions.append((res_x, res_y))
            bounds_list.append(src.bounds)
            crs_set.add(src.crs)

    if len(crs_set) > 1:
        raise ValueError("Input rasters must have the same CRS.")

    resolutions_array = np.array(resolutions)
    if resolution == "highest":
        target_res = resolutions_array.min(axis=0)
    elif resolution == "lowest":
        target_res = resolutions_array.max(axis=0)
    else:
        target_res = resolutions_array.mean(axis=0)
    if debug_logs: print(f'Resolution: {target_res}')

    temp_dir = tempfile.mkdtemp()
    tapped_paths = []

    # Loop 1: Tapping (if enabled)
    for in_path in input_image_paths:
        raster_name = os.path.splitext(os.path.basename(in_path))[0]
        with rasterio.open(in_path) as src:
            profile = src.profile.copy()

            if tap:
                res_x, res_y = target_res
                minx = np.floor(src.bounds.left / res_x) * res_x
                miny = np.floor(src.bounds.bottom / res_y) * res_y
                maxx = np.ceil(src.bounds.right / res_x) * res_x
                maxy = np.ceil(src.bounds.top / res_y) * res_y

                dst_width = int((maxx - minx) / res_x)
                dst_height = int((maxy - miny) / res_y)
                dst_transform = rasterio.transform.from_origin(minx, maxy, res_x, res_y)

                profile.update({
                    "height": dst_height,
                    "width": dst_width,
                    "transform": dst_transform
                })

                temp_path = os.path.join(temp_dir, f"{raster_name}_tapped.tif")
                tapped_paths.append(temp_path)

                src_windows = list(_create_windows(src.width, src.height, *window_size)) if window_size else [Window(0, 0, src.width, src.height)]
                dst_windows = list(_create_windows(dst_width, dst_height, *window_size)) if window_size else [Window(0, 0, dst_width, dst_height)]

                with rasterio.open(temp_path, "w", **profile) as dst:
                    for src_win, dst_win in zip(src_windows, dst_windows):
                        reproject(
                            source=rasterio.band(src, list(range(1, src.count + 1))),
                            destination=rasterio.band(dst, list(range(1, src.count + 1))),
                            src_transform=src.window_transform(src_win),
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=src.crs,
                            src_nodata=src.nodata,
                            dst_nodata=src.nodata,
                            resampling=Resampling[resampling_method],
                            src_window=src_win,
                            dst_window=dst_win
                        )
            else:
                tapped_paths.append(in_path)

    # Loop 2: Apply mask
    for in_path, out_path in zip(tapped_paths, output_image_paths):
        raster_name = os.path.splitext(os.path.basename(in_path))[0].replace('_tapped', '')
        if debug_logs: print(f'Masking: {raster_name}')
        with rasterio.open(in_path) as src:
            if split_mask_by_attribute:
                filtered_gdf = gdf[gdf[split_mask_by_attribute].str.strip() == raster_name.strip()]
                if filtered_gdf.empty:
                    if debug_logs: print(f"No matching features")
                    continue
                geometries = filtered_gdf.geometry.values
            else:
                geometries = gdf.geometry.values

            profile = src.profile.copy()

            if window_size: windows = list(_create_windows(src.width, src.height, *window_size))
            else: windows = [Window(0, 0, src.width, src.height)]

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with rasterio.open(out_path, "w", **profile) as dst:
                for window in windows:
                    data = src.read(window=window)
                    transform = src.window_transform(window)

                    mask_array = geometry_mask(
                        geometries,
                        out_shape=(data.shape[1], data.shape[2]),
                        transform=transform,
                        invert=True,
                        all_touched=include_touched_pixels
                    )

                    masked = np.where(mask_array, data, src.nodata)
                    dst.write(masked, window=window)

    # Cleanup
    if tap:
        shutil.rmtree(temp_dir)