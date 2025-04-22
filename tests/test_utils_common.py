# import pytest
#
# from spectralmatch.utils.utils_common import _get_image_metadata
#
#
# def test_get_image_metadata_valid_tiff(mocker):
#     mock_dataset = mocker.MagicMock()
#     mock_dataset.GetGeoTransform.return_value = (0, 1, 0, 500, 0, -1)
#     mock_dataset.GetProjection.return_value = "EPSG:4326"
#     mock_band = mocker.MagicMock()
#     mock_band.GetNoDataValue.return_value = -9999
#     mock_dataset.GetRasterBand.return_value = mock_band
#     mock_dataset.RasterCount = 1
#     mock_dataset.RasterXSize = 100
#     mock_dataset.RasterYSize = 100
#     mocker.patch(
#         "spectralmatch.utils.utils_common.gdal.Open", return_value=mock_dataset
#     )
#
#     transform, projection, nodata, bounds = _get_image_metadata("valid_image.tiff")
#
#     assert transform == (0, 1, 0, 500, 0, -1)
#     assert projection == "EPSG:4326"
#     assert nodata == -9999
#     assert bounds == {
#         "x_min": 0,
#         "y_min": 400,
#         "x_max": 100,
#         "y_max": 500,
#     }
#
#
# def test_get_image_metadata_invalid_file(mocker):
#     mocker.patch("spectralmatch.utils.utils_common.gdal.Open", return_value=None)
#
#     transform, projection, nodata, bounds = _get_image_metadata("invalid_image.tiff")
#
#     assert transform is None
#     assert projection is None
#     assert nodata is None
#     assert bounds is None
#
#
# def test_get_image_metadata_no_transform(mocker):
#     mock_dataset = mocker.MagicMock()
#     mock_dataset.GetGeoTransform.return_value = None
#     mock_dataset.GetProjection.return_value = "EPSG:4326"
#     mock_band = mocker.MagicMock()
#     mock_band.GetNoDataValue.return_value = None
#     mock_dataset.GetRasterBand.return_value = mock_band
#     mock_dataset.RasterCount = 1
#     mocker.patch(
#         "spectralmatch.utils.utils_common.gdal.Open", return_value=mock_dataset
#     )
#
#     transform, projection, nodata, bounds = _get_image_metadata(
#         "no_transform_image.tiff"
#     )
#
#     assert transform is None
#     assert projection == "EPSG:4326"
#     assert nodata is None
#     assert bounds is None
#
#
# @pytest.mark.skip(reason="This test is not working")
# def test_merge_rasters(mocker):
#     pass
