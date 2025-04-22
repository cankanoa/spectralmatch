import os

from osgeo import gdal
import rasterio
import warnings
from typing import List, Union, Optional

def _check_raster_requirements(
    input_image_paths,
    ) -> bool:

    print(f"Found {len(input_image_paths)} images")
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
    print("Input data checks passed: geotransform match, CRS match, band count match, nodata match")
    return True

def _get_nodata_value(
    input_image_paths: List[Union[str]],
    custom_nodata_value: Optional[float] = None,
    ) -> float | None:

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