import numpy as np
import rasterio
import pytest
import geopandas as gpd

from rasterio.transform import from_origin
from shapely.geometry import box


def create_dummy_raster(path, width=10, height=10, count=3, dtype='uint8', nodata=None, transform=None, crs='EPSG:4326', fill_value=100):
    if transform is None:
        transform = from_origin(0, 10, 1, 1)  # arbitrary 1x1 pixel size

    data = np.full((count, height, width), fill_value, dtype=dtype)

    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata
    ) as dst:
        dst.write(data)


def create_dummy_vector(path, bounds=(0, 0, 5, 5), crs="EPSG:4326"):
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(*bounds)], crs=crs)
    gdf.to_file(path)