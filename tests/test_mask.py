import numpy as np
import rasterio
import os
import pytest
import geopandas as gpd

from rasterio.transform import from_origin

from spectralmatch import (
    band_math,
    create_cloud_mask_with_omnicloudmask,
    process_raster_values_to_vector_polygons,
    threshold_raster,
    create_ndvi_raster,
)
from .utils_test import create_dummy_raster


@pytest.fixture
def dummy_multiband_raster(tmp_path):
    path = tmp_path / "input.tif"
    create_dummy_raster(path, width=5, height=5, count=2, fill_value=10)
    return path


@pytest.fixture
def dummy_rgbn_raster(tmp_path):
    path = tmp_path / "rgbn.tif"
    create_dummy_raster(path, width=128, height=128, count=3, fill_value=100)
    return path


@pytest.fixture
def dummy_red_nir_raster(tmp_path):
    path = tmp_path / "rgbn.tif"
    width, height = 32, 32
    transform = from_origin(0, 32, 1, 1)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=2,
        dtype="uint16",
        transform=transform,
        crs="EPSG:4326",
    ) as dst:
        dst.write(np.full((height, width), 1000, dtype="uint16"), 1)  # NIR
        dst.write(np.full((height, width), 500, dtype="uint16"), 2)  # Red
    return path


@pytest.fixture
def dummy_raster_for_vector(tmp_path):
    path = tmp_path / "input.tif"
    width, height = 16, 16
    transform = from_origin(0, 16, 1, 1)
    data = np.zeros((height, width), dtype="uint8")
    data[2:6, 2:6] = 1
    data[10:14, 10:14] = 2

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        transform=transform,
        crs="EPSG:4326",
    ) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def dummy_gradient_raster(tmp_path):
    path = tmp_path / "input.tif"
    data = np.tile(np.arange(16, dtype="uint8"), (16, 1))  # Horizontal gradient
    transform = from_origin(0, 16, 1, 1)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return path


# band_math
def test_band_math_basic(dummy_multiband_raster, tmp_path):
    input_path = str(dummy_multiband_raster)
    output_path = str(tmp_path / "output.tif")
    band_math(
        input_images=[input_path], output_images=[output_path], custom_math="b1 + b2"
    )

    with rasterio.open(output_path) as out:
        result = out.read(1)
        assert result.shape == (5, 5)
        assert np.all(result == 20)


def test_band_math_dtype(dummy_multiband_raster, tmp_path):
    input_path = str(dummy_multiband_raster)
    output_path = str(tmp_path / "typed.tif")
    band_math(
        input_images=[input_path],
        output_images=[output_path],
        custom_math="b1 * 2",
        custom_output_dtype="uint16",
    )

    with rasterio.open(output_path) as out:
        assert out.dtypes[0] == "uint16"


def test_band_math_nodata(dummy_multiband_raster, tmp_path):
    input_path = str(dummy_multiband_raster)
    output_path = str(tmp_path / "nodata.tif")
    band_math(
        input_images=[input_path],
        output_images=[output_path],
        custom_math="b1 + b2",
        custom_nodata_value=99,
    )

    with rasterio.open(output_path) as out:
        assert out.nodata == 99


# create_cloud_mask_with_omnicloudmask
def test_create_cloud_mask(dummy_rgbn_raster, tmp_path):
    input_path = str(dummy_rgbn_raster)
    output_path = str(tmp_path / "cloud_mask.tif")

    create_cloud_mask_with_omnicloudmask(
        input_images=[input_path],
        output_images=[output_path],
        red_band_index=1,
        green_band_index=2,
        nir_band_index=3,
        debug_logs=True,
        omnicloud_kwargs={"patch_size": 50, "patch_overlap": 20},
    )

    assert os.path.exists(output_path)
    with rasterio.open(output_path) as out:
        assert out.read(1).shape == (128, 128)


# create_ndvi_raster
def test_create_ndvi_raster_basic(dummy_red_nir_raster, tmp_path):
    input_path = str(dummy_red_nir_raster)
    output_path = str(tmp_path / "ndvi.tif")

    create_ndvi_raster(
        input_images=[input_path],
        output_images=[output_path],
        nir_band_index=1,
        red_band_index=2,
    )

    assert tmp_path.joinpath("ndvi.tif").exists()
    with rasterio.open(output_path) as ds:
        ndvi_data = ds.read(1)
        assert ndvi_data.shape == (32, 32)
        assert np.allclose(ndvi_data, (1000 - 500) / (1000 + 500), atol=1e-3)


# process_raster_values_to_vector_polygons
def test_process_raster_values_to_polygons_basic(dummy_raster_for_vector, tmp_path):
    input_path = str(dummy_raster_for_vector)
    output_path = str(tmp_path / "out.gpkg")

    process_raster_values_to_vector_polygons(
        input_images=[input_path],
        output_vectors=[output_path],
        extraction_expression="b1 > 0",
    )

    assert tmp_path.joinpath("out.gpkg").exists()
    gdf = gpd.read_file(output_path)
    assert not gdf.empty
    assert gdf.geometry.iloc[0].is_valid


def test_process_polygons_with_value_mapping_and_filter(
    dummy_raster_for_vector, tmp_path
):
    input_path = str(dummy_raster_for_vector)
    output_path = str(tmp_path / "filtered.gpkg")

    process_raster_values_to_vector_polygons(
        input_images=[input_path],
        output_vectors=[output_path],
        extraction_expression="b1 >= 2",
        filter_by_polygon_size="<50%",
        value_mapping={2: 5},
    )

    gdf = gpd.read_file(output_path)
    assert all(gdf.area > 4)


def test_process_polygons_with_buffer(dummy_raster_for_vector, tmp_path):
    input_path = str(dummy_raster_for_vector)
    output_path = str(tmp_path / "buffered.gpkg")

    process_raster_values_to_vector_polygons(
        input_images=[input_path],
        output_vectors=[output_path],
        extraction_expression="b1 == 1",
        polygon_buffer=0.5,
    )

    gdf = gpd.read_file(output_path)
    assert gdf.geometry.iloc[0].buffer(-0.5).area < gdf.geometry.iloc[0].area


def test_invalid_filter_by_polygon_size_raises(dummy_raster_for_vector, tmp_path):
    input_path = str(dummy_raster_for_vector)
    output_path = str(tmp_path / "error.gpkg")

    with pytest.raises(ValueError):
        process_raster_values_to_vector_polygons(
            input_images=[input_path],
            output_vectors=[output_path],
            extraction_expression="b1 > 0",
            filter_by_polygon_size="50%",  # Invalid
        )


# threshold_raster
def test_threshold_raster_basic(dummy_gradient_raster, tmp_path):
    input_path = str(dummy_gradient_raster)
    output_path = str(tmp_path / "out.tif")

    threshold_raster(
        input_images=[input_path],
        output_images=[output_path],
        threshold_math="b1 > 5",
    )

    with rasterio.open(output_path) as out:
        result = out.read(1)
        assert result.dtype == np.uint8
        assert np.all((result == 0) | (result == 1))
        assert np.sum(result) > 0


def test_threshold_raster_compound(dummy_gradient_raster, tmp_path):
    input_path = str(dummy_gradient_raster)
    output_path = str(tmp_path / "out_compound.tif")

    threshold_raster(
        input_images=[input_path],
        output_images=[output_path],
        threshold_math="(b1 > 5) & (b1 < 10)",
    )

    with rasterio.open(output_path) as out:
        result = out.read(1)
        assert np.all((result == 0) | (result == 1))
        assert np.any(result == 1)


def test_threshold_raster_percentile(dummy_gradient_raster, tmp_path):
    input_path = str(dummy_gradient_raster)
    output_path = str(tmp_path / "out_percentile.tif")

    threshold_raster(
        input_images=[input_path],
        output_images=[output_path],
        threshold_math="b1 > 95%b1",
    )

    with rasterio.open(output_path) as out:
        result = out.read(1)
        assert np.any(result == 1)
        assert np.all((result == 0) | (result == 1))
