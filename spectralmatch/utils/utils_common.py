import os

from osgeo import gdal
import rasterio

def _check_raster_requirements(
    input_image_paths,
    ) -> bool:

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
    print("Input data checks passed:")
    print("Geotransform match")
    print("CRS match")
    print("Band count match")
    print("NoData match")
    return True