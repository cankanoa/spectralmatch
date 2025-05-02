import warnings
import sys
import multiprocessing as mp
import rasterio

from typing import List, Optional, Union
from rasterio.windows import Window


def _check_raster_requirements(
    input_image_paths: list,
    debug_mode: bool,
    ) -> bool:
    """
    Validates a list of raster image paths to ensure they are compatible for processing.

    Args:
    input_image_paths (list[str]): Paths to input raster images.
    debug_mode (bool): If True, prints debug messages.

    Returns:
    bool: True if all input images meet geospatial and metadata consistency checks.

    Raises:
    ValueError: If any image lacks a geotransform, has a mismatched CRS, band count, or nodata value.
    """

    if debug_mode: print(f"Found {len(input_image_paths)} images")
    datasets = []
    for path in input_image_paths:
        data_in = rasterio.open(path)
        datasets.append(data_in)

    ref_crs = datasets[0].crs
    ref_count = datasets[0].count
    ref_nodata = [datasets[0].nodata] * ref_count if datasets[0].nodata is not None else [None] * ref_count

    for i, ds in enumerate(datasets):
        if ds.transform is None:
            raise ValueError(f"Fail: Image {i} has no geotransform.")
        if ds.crs != ref_crs:
            raise ValueError(f"Fail: Image {i} has different CRS.")
        if ds.count != ref_count:
            raise ValueError(f"Fail: Image {i} has {ds.count} bands; expected {ref_count}.")
        for b in range(ds.count):
            if ds.nodata != ref_nodata[b]:
                raise ValueError(f"Fail: Image {i}, band {b+1} has different nodata value.")
    if debug_mode: print("Input data checks passed: geotransform are present, CRS match, band count match, nodata match")
    return True


def _get_nodata_value(
    input_image_paths: List[Union[str]],
    custom_nodata_value: Optional[float] = None,
    ) -> float | None:
    """
    Determines the NoData value to use from a list of raster images or a custom override.

    Args:
    input_image_paths (List[str]): List of raster image paths.
    custom_nodata_value (float, optional): User-defined NoData value.

    Returns:
    float | None: The determined NoData value, or None if unavailable.

    Warnings:
    Emits a warning if a custom value overrides the image value or if no value is found.
    """

    try:
        with rasterio.open(input_image_paths[0]) as ds: image_nodata_value = ds.nodata
    except:
        image_nodata_value = None

    if custom_nodata_value is None and image_nodata_value is not None:
        return image_nodata_value

    if custom_nodata_value is not None:
        if image_nodata_value is not None and image_nodata_value != custom_nodata_value:
            warnings.warn("Image no data value has been overwritten by custom_nodata_value")
        return custom_nodata_value

    warnings.warn("Custom nodata value not set and could not get one from the first band so no nodata value will be used.")
    return None


def _create_windows(
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
    ):
    """
    Generates tiled windows across a raster based on specified dimensions.

    Args:
    width (int): Total width of the raster.
    height (int): Total height of the raster.
    tile_width (int): Width of each tile.
    tile_height (int): Height of each tile.

    Yields:
    rasterio.windows.Window: A window representing a tile's position and size.
    """

    for row_off in range(0, height, tile_height):
        for col_off in range(0, width, tile_width):
            win_width = min(tile_width, width - col_off)
            win_height = min(tile_height, height - row_off)
            yield Window(col_off, row_off, win_width, win_height)


def _choose_context(
    prefer_fork: bool = True
    ) -> mp.context.BaseContext:
    """
    Selects the optimal multiprocessing context based on platform and preference.

    Args:
    prefer_fork (bool): If True, prefers 'fork' context when available. Defaults to True.

    Returns:
    multiprocessing.context.BaseContext: The selected multiprocessing context.
    """

    if prefer_fork and sys.platform.startswith("linux"):
        return mp.get_context("fork")
    if prefer_fork and sys.platform == "darwin":
        try:
            return mp.get_context("fork")
        except ValueError:
            pass
    try:
        return mp.get_context("forkserver")
    except ValueError:
        return mp.get_context("spawn")