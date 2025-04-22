import pytest
from spectralmatch.utils.utils_common import _check_raster_requirements
from unittest.mock import MagicMock

def _mock_raster(crs, count, nodata, transform=(0, 1, 0, 0, 0, -1)):
    mock_ds = MagicMock()
    mock_ds.crs = crs
    mock_ds.count = count
    mock_ds.nodata = nodata
    mock_ds.transform = transform
    return mock_ds

def test_check_raster_requirements_valid(mocker):
    # All rasters have the same metadata
    mock_ds1 = _mock_raster("EPSG:4326", 3, -9999)
    mock_ds2 = _mock_raster("EPSG:4326", 3, -9999)
    mocker.patch("spectralmatch.utils.utils_common.rasterio.open", side_effect=[mock_ds1, mock_ds2])

    result = _check_raster_requirements(["img1.tif", "img2.tif"])
    assert result is True

def test_check_raster_requirements_different_crs(mocker):
    mock_ds1 = _mock_raster("EPSG:4326", 3, -9999)
    mock_ds2 = _mock_raster("EPSG:3857", 3, -9999)
    mocker.patch("spectralmatch.utils.utils_common.rasterio.open", side_effect=[mock_ds1, mock_ds2])

    with pytest.raises(ValueError, match="different CRS"):
        _check_raster_requirements(["img1.tif", "img2.tif"])

def test_check_raster_requirements_different_band_count(mocker):
    mock_ds1 = _mock_raster("EPSG:4326", 3, -9999)
    mock_ds2 = _mock_raster("EPSG:4326", 1, -9999)
    mocker.patch("spectralmatch.utils.utils_common.rasterio.open", side_effect=[mock_ds1, mock_ds2])

    with pytest.raises(ValueError, match="has 1 bands; expected 3"):
        _check_raster_requirements(["img1.tif", "img2.tif"])

def test_check_raster_requirements_missing_transform(mocker):
    mock_ds1 = _mock_raster("EPSG:4326", 3, -9999)
    mock_ds2 = _mock_raster("EPSG:4326", 3, -9999)
    mock_ds2.transform = None
    mocker.patch("spectralmatch.utils.utils_common.rasterio.open", side_effect=[mock_ds1, mock_ds2])

    with pytest.raises(ValueError, match="has no geotransform"):
        _check_raster_requirements(["img1.tif", "img2.tif"])

def test_check_raster_requirements_different_nodata(mocker):
    mock_ds1 = _mock_raster("EPSG:4326", 2, -9999)
    mock_ds2 = _mock_raster("EPSG:4326", 2, None)
    mocker.patch("spectralmatch.utils.utils_common.rasterio.open", side_effect=[mock_ds1, mock_ds2])

    with pytest.raises(ValueError, match="band 1 has different nodata value"):
        _check_raster_requirements(["img1.tif", "img2.tif"])