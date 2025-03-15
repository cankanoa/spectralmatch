import os

from osgeo import gdal


def _merge_rasters(input_array, output_image_folder, output_file_name="merge.tif"):

    output_path = os.path.join(output_image_folder, output_file_name)
    input_datasets = [gdal.Open(path) for path in input_array if gdal.Open(path)]
    gdal.Warp(
        output_path,
        input_datasets,
        format="GTiff",
    )

    print(f"Merged raster saved to: {output_path}")


def _get_image_metadata(input_image_path):
    """
    Get metadata of a TIFF image, including transform, projection, nodata, and bounds.

    Args:
    input_image_path (str): Path to the input image file.

    Returns:
    tuple: A tuple containing (transform, projection, nodata, bounds).
    """
    try:
        dataset = gdal.Open(input_image_path, gdal.GA_ReadOnly)
        if dataset is not None:
            # Get GeoTransform
            transform = dataset.GetGeoTransform()

            # Get Projection
            projection = dataset.GetProjection()

            # Get NoData value (assuming from the first band)
            nodata = (
                dataset.GetRasterBand(1).GetNoDataValue()
                if dataset.RasterCount > 0
                else None
            )

            # Calculate bounds
            if transform:
                x_min = transform[0]
                y_max = transform[3]
                x_max = x_min + (dataset.RasterXSize * transform[1])
                y_min = y_max + (dataset.RasterYSize * transform[5])
                bounds = {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                }
            else:
                bounds = None

            dataset = None  # Close the dataset

            return transform, projection, nodata, bounds
        else:
            print(f"Could not open the file: {input_image_path}")
    except Exception as e:
        print(f"Error processing {input_image_path}: {e}")
    return None, None, None, None
