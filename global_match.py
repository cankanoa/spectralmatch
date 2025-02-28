import sys

import numpy as np
from osgeo import gdal
from scipy.optimize import least_squares
import os
np.set_printoptions(
    suppress=True,
    precision=3,
    linewidth=300,
    formatter={'float_kind':lambda x: f"{x: .3f}"}
)
gdal.UseExceptions()

def get_image_metadata(input_image_path):
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
            nodata = dataset.GetRasterBand(1).GetNoDataValue() if dataset.RasterCount > 0 else None

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

def find_overlaps(image_bounds_dict):
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
def calculate_image_stats(num_bands, input_image_path_i, input_image_path_j, id_i, id_j, bound_i, bound_j, nodata_i, nodata_j):
    """
Calculate overlap statistics (mean, std, and size) for two overlapping images,
as well as their individual statistics, while excluding NoData values.

Args:
num_bands (int): Number of bands in the images.
input_image_path_i (str): Path to the first image.
input_image_path_j (str): Path to the second image.
id_i (int): ID of the first image.
id_j (int): ID of the second image.
bound_i (dict): Bounds of the first image in the format {"x_min", "x_max", "y_min", "y_max"}.
bound_j (dict): Bounds of the second image in the format {"x_min", "x_max", "y_min", "y_max"}.
nodata_i (float): The NoData value for the first image.
nodata_j (float): The NoData value for the second image.

Returns:
tuple: A tuple containing:
- overlap_stat: Dictionary of overlap statistics in the format:
{id_i: {id_j: {band: {'mean': value, 'std': value, 'size': value}}},
id_j: {id_i: {band: {'mean': value, 'std': value, 'size': value}}}}
- whole_stats: Dictionary of whole image statistics in the format:
{id_i: {band: {'mean': value, 'std': value, 'size': value}}}
    """
    from osgeo import gdal
    import numpy as np

    # Initialize the result dictionaries
    overlap_stat = {id_i: {id_j: {}}, id_j: {id_i: {}}}
    whole_stats = {id_i: {}, id_j: {}}

    # Open both images
    dataset_i = gdal.Open(input_image_path_i)
    dataset_j = gdal.Open(input_image_path_j)

    if not dataset_i or not dataset_j:
        raise RuntimeError("Failed to open one or both input images.")

    geo_transform_i = dataset_i.GetGeoTransform()
    geo_transform_j = dataset_j.GetGeoTransform()

    # Calculate overlap bounds
    x_min_overlap = max(bound_i["x_min"], bound_j["x_min"])
    x_max_overlap = min(bound_i["x_max"], bound_j["x_max"])
    y_min_overlap = max(bound_i["y_min"], bound_j["y_min"])
    y_max_overlap = min(bound_i["y_max"], bound_j["y_max"])

    if x_min_overlap >= x_max_overlap or y_min_overlap >= y_max_overlap:
        return overlap_stat, whole_stats

    # Calculate pixel ranges for the overlaps
    col_min_i = int((x_min_overlap - geo_transform_i[0]) / geo_transform_i[1])
    col_max_i = int((x_max_overlap - geo_transform_i[0]) / geo_transform_i[1])
    row_min_i = int((y_max_overlap - geo_transform_i[3]) / geo_transform_i[5])
    row_max_i = int((y_min_overlap - geo_transform_i[3]) / geo_transform_i[5])

    col_min_j = int((x_min_overlap - geo_transform_j[0]) / geo_transform_j[1])
    col_max_j = int((x_max_overlap - geo_transform_j[0]) / geo_transform_j[1])
    row_min_j = int((y_max_overlap - geo_transform_j[3]) / geo_transform_j[5])
    row_max_j = int((y_min_overlap - geo_transform_j[3]) / geo_transform_j[5])

    # Ensure overlap shapes align
    overlap_rows = min(row_max_i - row_min_i, row_max_j - row_min_j)
    overlap_cols = min(col_max_i - col_min_i, col_max_j - col_min_j)

    row_max_i, row_max_j = row_min_i + overlap_rows, row_min_j + overlap_rows
    col_max_i, col_max_j = col_min_i + overlap_cols, col_min_j + overlap_cols

    # Process each band
    for band_idx in range(num_bands):
        band_data_i = dataset_i.GetRasterBand(band_idx + 1).ReadAsArray()
        band_data_j = dataset_j.GetRasterBand(band_idx + 1).ReadAsArray()

        mask_i = band_data_i != nodata_i
        mask_j = band_data_j != nodata_j

        # Calculate whole image statistics for each image
        valid_data_i = band_data_i[mask_i]
        valid_data_j = band_data_j[mask_j]
        whole_stats[id_i][band_idx] = {
            "mean": np.mean(valid_data_i) if valid_data_i.size > 0 else 0,
            "std": np.std(valid_data_i) if valid_data_i.size > 0 else 0,
            "size": valid_data_i.size,
        }
        whole_stats[id_j][band_idx] = {
            "mean": np.mean(valid_data_j) if valid_data_j.size > 0 else 0,
            "std": np.std(valid_data_j) if valid_data_j.size > 0 else 0,
            "size": valid_data_j.size,
        }

        # Slice data and masks for overlaps
        overlap_data_i = band_data_i[row_min_i:row_max_i, col_min_i:col_max_i]
        overlap_data_j = band_data_j[row_min_j:row_max_j, col_min_j:col_max_j]
        overlap_mask_i = mask_i[row_min_i:row_max_i, col_min_i:col_max_i]
        overlap_mask_j = mask_j[row_min_j:row_max_j, col_min_j:col_max_j]

        # Combine masks and calculate overlap data
        overlap_mask = overlap_mask_i & overlap_mask_j
        overlap_data_combined_i = overlap_data_i[overlap_mask]
        overlap_data_combined_j = overlap_data_j[overlap_mask]

        # Calculate overlap statistics
        mean1 = np.mean(overlap_data_combined_i) if overlap_data_combined_i.size > 0 else 0
        std1 = np.std(overlap_data_combined_i) if overlap_data_combined_i.size > 0 else 0
        mean2 = np.mean(overlap_data_combined_j) if overlap_data_combined_j.size > 0 else 0
        std2 = np.std(overlap_data_combined_j) if overlap_data_combined_j.size > 0 else 0
        overlap_size = overlap_mask.sum()

        overlap_stat[id_i][id_j][band_idx] = {
            "mean": mean1,
            "std": std1,
            "size": overlap_size,
        }
        overlap_stat[id_j][id_i][band_idx] = {
            "mean": mean2,
            "std": std2,
            "size": overlap_size,
        }

    dataset_i = None
    dataset_j = None

    return overlap_stat, whole_stats

import os
import numpy as np
from osgeo import gdal

def append_band_to_tif(
        band_array: np.ndarray,
        transform: tuple,
        projection: str,
        output_path: str,
        nodata_value: float,
        band_index: int,
        total_bands: int,
        dtype=gdal.GDT_Int16
):
    """
Appends (or writes) a single band to a GeoTIFF file.

1) If the TIFF file does not exist, create it with 'total_bands' bands.
- Then write the band_array into band_index.
2) If the TIFF file exists, open in update mode and write the band_array
into band_index (1-based).

No reading of existing pixel data is done.

Args:
band_array    : 2D NumPy array with shape (rows, cols).
transform     : (gt0, gt1, gt2, gt3, gt4, gt5) – same as GDAL GeoTransform.
projection    : WKT string describing projection (same as ds.GetProjection()).
output_path   : Path to the output GeoTIFF.
nodata_value  : Value to set as NoData.
band_index    : 1-based index of the band to write (1 <= band_index <= total_bands).
total_bands   : How many bands in the final dataset (must be >= band_index).
dtype         : gdal data type, e.g. gdal.GDT_Float32, gdal.GDT_Int16, etc.
    """
    rows, cols = band_array.shape

    # 1. If file does not exist, create it
    if not os.path.exists(output_path):
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(
            output_path,
            cols,
            rows,
            total_bands,
            dtype
        )
        # Set georeferencing only once (on creation)
        out_ds.SetGeoTransform(transform)
        out_ds.SetProjection(projection)

        # Initialize all bands to the nodata (optional, if you want them pre-filled)
        for b_i in range(1, total_bands+1):
            band_tmp = out_ds.GetRasterBand(b_i)
            band_tmp.Fill(nodata_value)
            band_tmp.SetNoDataValue(nodata_value)

        # Now write our band
        out_band = out_ds.GetRasterBand(band_index)
        out_band.WriteArray(band_array)
        out_band.SetNoDataValue(nodata_value)

        out_ds.FlushCache()
        out_ds = None

    else:
        # 2. If file exists, open in UPDATE mode
        out_ds = gdal.Open(output_path, gdal.GA_Update)
        if out_ds is None:
            raise RuntimeError(f"Could not open {output_path} in update mode.")

        # We assume it already has the correct 'total_bands' allocated
        if band_index > out_ds.RasterCount:
            raise ValueError(
                f"Band index {band_index} exceeds "
                f"the number of bands ({out_ds.RasterCount}) in {output_path}."
            )

        # Write the band
        out_band = out_ds.GetRasterBand(band_index)
        out_band.WriteArray(band_array)
        out_band.SetNoDataValue(nodata_value)

        out_ds.FlushCache()
        out_ds = None  # close

    print(f"Appended band {band_index} to {output_path}.")

def save_multiband_as_geotiff(array, geo_transform, projection, path, nodata_values):
    driver = gdal.GetDriverByName("GTiff")
    num_bands, rows, cols = array.shape
    out_ds = driver.Create(path, cols, rows, num_bands, gdal.GDT_Int16)
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)

    for i in range(num_bands):
        out_band = out_ds.GetRasterBand(i + 1)
        out_band.WriteArray(array[i])
        if nodata_values is not None:
            out_band.SetNoDataValue(nodata_values)

    out_ds.FlushCache()

def merge_rasters(input_array, output_image_folder, output_file_name="merge.tif"):

    output_path = os.path.join(output_image_folder, output_file_name)
    input_datasets = [gdal.Open(path) for path in input_array if gdal.Open(path)]
    gdal.Warp(
        output_path,
        input_datasets,
        format='GTiff',
    )

    print(f"Merged raster saved to: {output_path}")

def process_global_histogram_matching(input_image_paths_array, output_image_folder, output_global_basename, custom_mean_factor, custom_std_factor):
    # ---------------------------------------- Calculating statistics
    print('-------------------- Calculating statistics')
    num_bands = gdal.Open(input_image_paths_array[0], gdal.GA_ReadOnly).RasterCount
    num_images = len(input_image_paths_array)

    all_transforms = {}
    all_projections = {}
    all_nodata = {}
    all_bounds = {}
    for idx, input_image_path in enumerate(input_image_paths_array, start=0):
        all_transforms[idx], all_projections[idx], all_nodata[idx], all_bounds[idx] = get_image_metadata(input_image_path)

    overlapping_pairs = find_overlaps(all_bounds)

    all_overlap_stats = {}
    all_whole_stats = {}
    for id_i, id_j in overlapping_pairs:
        current_overlap_stats, current_whole_stats = calculate_image_stats(num_bands, input_image_paths_array[id_i], input_image_paths_array[id_j], id_i, id_j, all_bounds[id_i], all_bounds[id_j], all_nodata[id_i], all_nodata[id_j])

        all_overlap_stats.update({key_i: {**all_overlap_stats.get(key_i, {}), **{key_j: {**all_overlap_stats.get(key_i, {}).get(key_j, {}), **stats} for key_j, stats in value.items()}} for key_i, value in current_overlap_stats.items()})

        all_whole_stats.update(current_whole_stats)

    # ---------------------------------------- Model building and adjustment
    print('-------------------- Building Model and Applying Adjustments')

    # Prepare a 3D array to hold the final a/b parameters per band:
    #   shape: (num_bands, 2*num_images, 1)
    all_adjustment_params = np.zeros((num_bands, 2 * num_images, 1), dtype=float)

    for band_idx in range(num_bands):
        print(f"Processing band {band_idx + 1}/{num_bands}:")

        constraint_matrix = []
        observed_values_vector = []
        total_overlap_pixels = 0

        # We'll keep track of which pairs (i,j) we actually used, for printing
        overlap_pairs = []

        for i in range(num_images):
            for j in range(num_images):
                if i < j and all_overlap_stats.get(i, {}).get(j) is not None:

                    overlap_size = all_overlap_stats[i][j][band_idx]["size"]

                    # We'll gather the global (whole) stats for images i and j:
                    mean_1 = all_overlap_stats[i][j][band_idx]["mean"]
                    std_1  = all_overlap_stats[i][j][band_idx]["std"]
                    mean_2 = all_overlap_stats[j][i][band_idx]["mean"]
                    std_2  = all_overlap_stats[j][i][band_idx]["std"]

                    print( f"\tOverlap({i}-{j}):", end="")
                    print('\t', f'size: {overlap_size}px, mean:{mean_1:.2f} vs {mean_2:.2f}, std:{std_1:.2f} vs {std_2:.2f}')
                    overlap_pairs.append((i, j))
                    total_overlap_pixels += overlap_size

                    # mean difference: a_i * M_i + b_i - (a_j * M_j + b_j) = 0
                    # std difference: a_i * V_i - a_j * V_j = 0
                    num_params = 2 * num_images

                    # mean difference row
                    mean_row = [0]*num_params
                    mean_row[2*i] = mean_1
                    mean_row[2*i+1] = 1
                    mean_row[2*j] = -mean_2
                    mean_row[2*j+1] = -1

                    # std difference row
                    std_row = [0]*num_params
                    std_row[2*i] = std_1
                    std_row[2*j] = -std_2

                    # Apply overlap weight (p_ij = s_ij)
                    mean_row = [val * overlap_size * custom_mean_factor for val in mean_row]
                    std_row = [val * overlap_size * custom_std_factor for val in std_row]

                    # Observed values (targets) are 0 for these constraints
                    observed_values_vector.append(0)  # mean diff
                    observed_values_vector.append(0)  # std diff

                    constraint_matrix.append(mean_row)
                    constraint_matrix.append(std_row)

        if total_overlap_pixels == 0:
            pjj = 1.0
        else:
            pjj = total_overlap_pixels / (2.0 * num_images)

        # For each image, we want to keep its band-wide mean & std close to original
        #    mean constraint: a_j * M_j + b_j = M_j
        #    std constraint:  a_j * V_j = V_j
        for img_idx in range(num_images):
            Mj = all_whole_stats[img_idx][band_idx]["mean"]
            Vj = all_whole_stats[img_idx][band_idx]["std"]

            # mean constraint row
            mean_row = [0]*(2*num_images)
            mean_row[2*img_idx] = Mj
            mean_row[2*img_idx+1] = 1.0
            # we want: a_j*M_j + b_j - M_j = 0 => observed = M_j
            mean_obs = Mj

            # std constraint row
            std_row = [0]*(2*num_images)
            std_row[2*img_idx] = Vj
            # we want: a_j*V_j - V_j = 0 => observed = V_j
            std_obs = Vj

            # Weight these rows by p_jj
            mean_row = [val * pjj for val in mean_row]
            std_row = [val * pjj for val in std_row]

            mean_obs *= pjj
            std_obs *= pjj

            constraint_matrix.append(mean_row)
            observed_values_vector = np.append(observed_values_vector, mean_obs)

            constraint_matrix.append(std_row)
            observed_values_vector = np.append(observed_values_vector, std_obs)

        # ---------------------------------------- Model building
        if len(constraint_matrix) > 0:
            constraint_matrix = np.array(constraint_matrix)
            observed_values_vector = np.array(observed_values_vector)

            def residuals(params):
                return constraint_matrix @ params - observed_values_vector

            initial_params = [1.0, 0.0] * num_images
            result = least_squares(residuals, initial_params)
            adjustment_params = result.x.reshape((2 * num_images, 1))
        else:
            print(f"No overlaps for band {band_idx+1}")
            adjustment_params = np.tile([1.0, 0.0], (num_images, 1))

        all_adjustment_params[band_idx] = adjustment_params

        # ---------------------------------------- Print info
        print(f"Shape: constraint_matrix: {constraint_matrix.shape}, adjustment_params: {adjustment_params.shape}, observed_values_vector: {observed_values_vector.shape}")
        print("constraint_matrix with labels:")
        # np.savetxt(sys.stdout, constraint_matrix, fmt="%16.3f")

        row_labels = []
        overlap_count = len(overlap_pairs)  # You must have recorded overlaps somewhere

        # Add two labels per overlap pair
        for (i, j) in overlap_pairs:
            row_labels.append(f"Overlap({i}-{j}) Mean Diff")
            row_labels.append(f"Overlap({i}-{j}) Std Diff")

        # Then add two labels per image for mean/std constraints
        for img_idx in range(num_images):
            row_labels.append(f"Image {img_idx} Mean Cnstr")
            row_labels.append(f"Image {img_idx} Std Cnstr")

        # Now row_labels should have exactly constraint_matrix.shape[0] elements

        # Print column labels as before
        num_params = 2 * num_images
        col_labels = []
        for i in range(num_images):
            col_labels.append(f"a{i}")
            col_labels.append(f"b{i}")

        header = " " * 24  # extra space for row label
        for lbl in col_labels:
            header += f"{lbl:>18}"
        print(header)

        # Print each row with its label
        for row_label, row in zip(row_labels, constraint_matrix):
            line = f"{row_label:>24}"  # adjust the width as needed
            for val in row:
                line += f"{val:18.3f}"
            print(line)

        print('adjustment_params:')
        np.savetxt(sys.stdout, adjustment_params, fmt="%18.3f")
        print('observed_values_vector:')
        np.savetxt(sys.stdout, observed_values_vector, fmt="%18.3f")


    print('-------------------- Apply adjustments and saving results')
    output_path_array = []
    for img_idx in range(num_images):
        # for img_idx in [2]:
        adjusted_bands = []
        dataset = gdal.Open(input_image_paths_array[img_idx], gdal.GA_ReadOnly)
        print('open raster')
        if not dataset:
            print(f"Error: Could not open file {input_image_paths_array[img_idx]}")
            continue

        # adjusted_bands_array = np.stack(adjusted_bands, axis=0)
        input_filename = os.path.basename(input_image_paths_array[img_idx])
        output_filename = os.path.splitext(input_filename)[0] + output_global_basename + ".tif"
        os.makedirs(os.path.join(output_image_folder, 'images'), exist_ok=True)
        output_path = os.path.join(output_image_folder, 'images', output_filename)
        output_path_array.append(output_path)

        for band_idx in range(num_bands):
            raw_band = dataset.GetRasterBand(band_idx + 1)
            band = raw_band.ReadAsArray()
            input_dtype = raw_band.DataType

            if band is None:
                print(f"Error: Could not access band {band_idx + 1} in file {input_image_paths_array[img_idx]}")
                continue
            mask = band != all_nodata[img_idx]
            a = all_adjustment_params[band_idx, 2 * img_idx, 0]
            b = all_adjustment_params[band_idx, 2 * img_idx + 1, 0]

            adjusted_band = np.where(mask, a * band + b, band)
            # adjusted_bands.append(np.where(mask, a * band + b, band))

            append_band_to_tif(
                adjusted_band,
                all_transforms[img_idx],
                all_projections[img_idx],
                output_path,
                all_nodata[img_idx],
                band_idx + 1,
                total_bands=num_bands,
                # dtype=input_dtype,
            )
        dataset = None

        # save_multiband_as_geotiff(
        #     adjusted_bands_array,
        #     all_transforms[img_idx],
        #     all_projections[img_idx],
        #     output_path,
        #     all_nodata[img_idx],
        # )

        print(f"Saved file {img_idx} to: {output_path}")
    # ---------------------------------------- Merge rasters
    print('-------------------- Merging rasters and saving result')
    merge_rasters(output_path_array, output_image_folder, output_file_name=f"Merged{output_global_basename}.tif")


# # ---------------------------------------- Call function
# input_image_paths_array = [
#     # "/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFrom20182019Lidar.tif",
#     # "/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211800-M1BS-016445319010_01_P004_FLAASH_OrthoFrom20182019Lidar.tif",
#     # "/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211801-M1BS-016445319010_01_P005_FLAASH_PuuSubset_OrthoFrom20182019Lidar.tif",
#     # '/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211840-M1BS-016445318010_01_P015_FLAASH_OrthoFrom20182019Lidar.tif',
#     # '/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211841-M1BS-016445318010_01_P016_FLAASH_OrthoFrom20182019Lidar.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/3subset.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/4subset.tif',
#     '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu3.tif',
#     '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu4.tif',
#     '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu5.tif',
#     '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu15.tif',
#     '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu16.tif',
#
# ]
# output_image_folder = "/Users/kanoalindiwe/Downloads/temp/"
# output_global_basename = "_GlobalHistMatch_goodWholeMea3nNodata0"
# custom_mean_factor = 3 # Defualt 1
# custom_std_factor = 1 # Defualt 1
# process_global_histogram_matching(input_image_paths_array, output_image_folder, output_global_basename, custom_mean_factor, custom_std_factor)