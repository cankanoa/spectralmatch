
import numpy as np
import rasterio
import fiona

from spectralmatch.utils.utils_io import create_windows
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.transform import rowcol
from rasterio.features import geometry_mask
from osgeo import ogr

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


def _rasterize_mask(
    vector_path,
    image_name,
    ref_src
    ):
    if vector_path is None:
        return None

    gpkg = ogr.Open(vector_path)
    if gpkg is None:
        raise FileNotFoundError(f"Could not open vector file: {vector_path}")

    layer = gpkg.GetLayer(0)

    features = []
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom and not geom.IsEmpty():
            cleaned = geom.Clone()
            if not cleaned.IsValid():
                cleaned = cleaned.MakeValid()
            if cleaned and not cleaned.IsEmpty():
                features.append((cleaned.ExportToWkt(), 1))

    if not features:
        return None

    shapes = [(ogr.CreateGeometryFromWkt(wkt), value) for wkt, value in features]

    mask = rasterize(
        [(g, v) for g, v in shapes],
        out_shape=(ref_src.height, ref_src.width),
        transform=ref_src.transform,
        fill=0,
        dtype='uint8'
    )

    return mask

def _mask_tile(src, geoms, window):
    transform = src.window_transform(window)
    shape = (window.height, window.width)

    mask_arr = geometry_mask(
        geometries=geoms,
        transform=transform,
        invert=True,
        out_shape=shape
    )

    tile = src.read(window=window)
    return tile * mask_arr.astype(tile.dtype)

def _calculate_overlap_stats(
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
    tile_width_and_height_tuple: tuple = None,
    debug_mode=False,
    ):
    stats = {id_i: {id_j: {}}, id_j: {id_i: {}}}

    with rasterio.open(input_image_path_i) as src_i, rasterio.open(input_image_path_j) as src_j:
        if vector_mask_path:
            with fiona.open(vector_mask_path, "r") as vector:
                geoms = [feat["geometry"] for feat in vector]
        else:
            geoms = None

        transform_i = src_i.transform
        transform_j = src_j.transform

        x_min = max(bound_i["x_min"], bound_j["x_min"])
        x_max = min(bound_i["x_max"], bound_j["x_max"])
        y_min = max(bound_i["y_min"], bound_j["y_min"])
        y_max = min(bound_i["y_max"], bound_j["y_max"])

        if debug_mode: print(f"Overlap bounds: x: {x_min:.2f} to {x_max:.2f}, y: {y_min:.2f} to {y_max:.2f}")

        if x_min >= x_max or y_min >= y_max:
            return stats

        row_min_i, col_min_i = rowcol(transform_i, x_min, y_max)
        row_max_i, col_max_i = rowcol(transform_i, x_max, y_min)
        row_min_j, col_min_j = rowcol(transform_j, x_min, y_max)
        row_max_j, col_max_j = rowcol(transform_j, x_max, y_min)

        height = min(row_max_i - row_min_i, row_max_j - row_min_j)
        width = min(col_max_i - col_min_i, col_max_j - col_min_j)

        if debug_mode:
            print(f"Pixel window (i): cols: {col_min_i} to {col_max_i} ({col_max_i - col_min_i}), rows: {row_min_i} to {row_max_i} ({row_max_i - row_min_i})")
            print(f"Pixel window (j): cols: {col_min_j} to {col_max_j} ({col_max_j - col_min_j}), rows: {row_min_j} to {row_max_j} ({row_max_j - row_min_j})")

        for band in range(num_bands):
            if tile_width_and_height_tuple:
                windows = create_windows(width, height, tile_width_and_height_tuple[0], tile_width_and_height_tuple[1])
            else:
                windows = [Window(0, 0, width, height)]
            windows = _adjust_size_of_tiles_to_fit_bounds(windows, width, height)

            combined_i = []
            combined_j = []

            for win in windows:
                offset_to_window_i = Window(col_min_i + win.col_off, row_min_i + win.row_off, win.width, win.height)
                offset_to_window_j = Window(col_min_j + win.col_off, row_min_j + win.row_off, win.width, win.height)

                block_i = src_i.read(band + 1, window=offset_to_window_i)
                block_j = src_j.read(band + 1, window=offset_to_window_j)

                if geoms:
                    transform_i_win = src_i.window_transform(offset_to_window_i)
                    transform_j_win = src_j.window_transform(offset_to_window_j)

                    mask_i = geometry_mask(geoms, transform=transform_i_win, invert=True,
                                           out_shape=(int(offset_to_window_i.height), int(offset_to_window_i.width)))
                    mask_j = geometry_mask(geoms, transform=transform_j_win, invert=True,
                                           out_shape=(int(offset_to_window_j.height), int(offset_to_window_j.width)))

                    block_i[~mask_i] = nodata_i
                    block_j[~mask_j] = nodata_j

                valid = (block_i != nodata_i) & (block_j != nodata_j)
                if np.any(valid):
                    combined_i.append(block_i[valid])
                    combined_j.append(block_j[valid])

            v_i = np.concatenate(combined_i) if combined_i else np.array([])
            v_j = np.concatenate(combined_j) if combined_j else np.array([])

            stats[id_i][id_j][band] = {
                "mean": float(np.mean(v_i)) if v_i.size else 0,
                "std": float(np.std(v_i)) if v_i.size else 0,
                "size": int(v_i.size),
            }
            stats[id_j][id_i][band] = {
                "mean": float(np.mean(v_j)) if v_j.size else 0,
                "std": float(np.std(v_j)) if v_j.size else 0,
                "size": int(v_j.size),
            }
    return stats

def _adjust_size_of_tiles_to_fit_bounds(windows, max_width, max_height):
    """Ensure no window extends beyond the target width/height (overlap bounds)."""
    adjusted_windows = []
    for win in windows:
        new_width = min(win.width, max_width - win.col_off)
        new_height = min(win.height, max_height - win.row_off)

        if new_width > 0 and new_height > 0:
            adjusted_windows.append(Window(win.col_off, win.row_off, new_width, new_height))

    return adjusted_windows


def _calculate_whole_stats(
    input_image_path,
    nodata,
    num_bands,
    image_id,
    vector_mask_path=None,
    tile_width_and_height_tuple: tuple = None,
    ):
    stats = {image_id: {}}

    with rasterio.open(input_image_path) as data:
        # Load geometries once if needed
        if vector_mask_path:
            with fiona.open(vector_mask_path, "r") as vector:
                geoms = [feat["geometry"] for feat in vector]
        else:
            geoms = None

        for band_idx in range(num_bands):
            mean = 0.0
            M2 = 0.0
            count = 0

            if tile_width_and_height_tuple:
                windows = create_windows(data.width, data.height, tile_width_and_height_tuple[0], tile_width_and_height_tuple[1])
            else:
                windows = [Window(0, 0, data.width, data.height)]

            for win in windows:
                block = data.read(band_idx + 1, window=win)

                if geoms:
                    transform = data.window_transform(win)
                    mask = geometry_mask(
                        geoms,
                        transform=transform,
                        invert=True,
                        out_shape=(int(win.height), int(win.width))
                    )
                    block[~mask] = nodata

                valid = block != nodata
                values = block[valid]

                for x in values.ravel():
                    count += 1
                    delta = x - mean
                    mean += delta / count
                    M2 += delta * (x - mean)

            if count > 1:
                variance = M2 / (count - 1)
                std = np.sqrt(variance)
            else:
                mean = 0.0
                std = 0.0

            stats[image_id][band_idx] = {
                "mean": float(mean),
                "std": float(std),
                "size": count,
            }

    return stats