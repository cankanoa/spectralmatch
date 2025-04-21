import os
import gc
import sys

import numpy as np
from scipy.optimize import least_squares
from typing import Tuple, List, Optional, Literal
import rasterio
from typing import Literal

from spectralmatch.utils.utils_local import _get_bounding_rectangle
from spectralmatch.utils.utils_local import _compute_block_size
from spectralmatch.utils.utils_local import _compute_blocks
from spectralmatch.utils.utils_local import _weighted_bilinear_interpolation
from spectralmatch.utils.utils_local import _download_block_map
from spectralmatch.utils.utils_local import _apply_gamma_correction

from spectralmatch.utils.utils_common import _get_image_metadata

from spectralmatch.utils.utils_global import _find_overlaps
from spectralmatch.utils.utils_global import _calculate_overlap_stats, _calculate_whole_stats
from spectralmatch.utils.utils_io import create_windows
from rasterio.windows import Window

def global_match(
    input_image_paths_array,
    output_image_folder,
    custom_mean_factor=1,
    custom_std_factor=1,
    output_global_basename='_global',
    vector_mask_path=None,
    tile_width_and_height_tuple: tuple = None,
    debug_mode=False,
    ):
    """
    Adjusts global histograms of input images for seamless stitching by calculating
    statistics from overlapping regions and applying adjustments.

    The `global_histogram_match` function is designed to process a list of input
    images and adjust their histograms such that the statistical properties (mean
    and standard deviation) across overlapping regions are consistent while
    preserving global properties of each image. This function uses image metadata,
    overlap statistics, and optimization techniques to compute adjustment
    parameters. The output consists of adjusted images written to the specified
    output folder.

    Args:
    input_image_paths_array (list[str]): List of file paths to the input images
    to be processed.
    output_image_folder (str): Directory where the processed images will be
    saved.
    output_global_basename (str): Shared base name to use when saving output
    images. Outputs will be saved with this base name and additional suffix
    for each image.
    custom_mean_factor (float): Weight scaling factor for adjusting the
    difference in mean statistics across overlapping regions.
    custom_std_factor (float): Weight scaling factor for adjusting the
    difference in standard deviation statistics across overlapping regions.

    Raises:
    ValueError: If input image paths are invalid or processing fails due to
    inconsistent metadata or missing overlaps.
    TypeError: If arguments are of incorrect type or contain incompatible
    data formats.
    RuntimeError: If an issue occurs during processing, such as failure in
    optimization or reading image data.
    """
    print("----------Starting Global Matching")

    # ---------------------------------------- Calculating statistics
    print("-------------------- Calculating statistics")
    with rasterio.open(input_image_paths_array[0]) as src: num_bands = src.count
    num_images = len(input_image_paths_array)

    all_transforms = {}
    all_projections = {}
    all_nodata = {}
    all_bounds = {}
    for idx, input_image_path in enumerate(input_image_paths_array, start=0):
        all_transforms[idx], all_projections[idx], all_nodata[idx], all_bounds[idx] = (
            _get_image_metadata(input_image_path)
        )

    overlapping_pairs = _find_overlaps(all_bounds)

    all_overlap_stats = {}
    all_whole_stats = {}
    for id_i, id_j in overlapping_pairs:
        current_overlap_stats = _calculate_overlap_stats(
            num_bands,
            input_image_paths_array[id_i],
            input_image_paths_array[id_j],
            id_i,
            id_j,
            all_bounds[id_i],
            all_bounds[id_j],
            all_nodata[id_i],
            all_nodata[id_j],
            vector_mask_path=vector_mask_path,
            tile_width_and_height_tuple=tile_width_and_height_tuple,
            debug_mode=debug_mode,
        )
        all_overlap_stats.update({
            key_i: {
                **all_overlap_stats.get(key_i, {}),
                **{key_j: {**all_overlap_stats.get(key_i, {}).get(key_j, {}),**stats}
                for key_j, stats in value.items()}
                }
            for key_i, value in current_overlap_stats.items()})

    for idx, input_image_path in enumerate(input_image_paths_array, start=0):
        current_whole_stats = _calculate_whole_stats(
            input_image_path=input_image_path,
            nodata=all_nodata[idx],
            num_bands=num_bands,
            image_id=idx,
            vector_mask_path=vector_mask_path,
            tile_width_and_height_tuple=tile_width_and_height_tuple
            )
        all_whole_stats.update(current_whole_stats)

    # ---------------------------------------- Model building and adjustment
    print("-------------------- Building Model and Applying Adjustments")

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
                    std_1 = all_overlap_stats[i][j][band_idx]["std"]
                    mean_2 = all_overlap_stats[j][i][band_idx]["mean"]
                    std_2 = all_overlap_stats[j][i][band_idx]["std"]

                    print(f"\tOverlap({i}-{j}):", end="")
                    print(
                        "\t",
                        f"size: {overlap_size}px, mean:{mean_1:.2f} vs {mean_2:.2f}, std:{std_1:.2f} vs {std_2:.2f}",
                    )
                    overlap_pairs.append((i, j))
                    total_overlap_pixels += overlap_size

                    # mean difference: a_i * M_i + b_i - (a_j * M_j + b_j) = 0
                    # std difference: a_i * V_i - a_j * V_j = 0
                    num_params = 2 * num_images

                    # mean difference row
                    mean_row = [0] * num_params
                    mean_row[2 * i] = mean_1
                    mean_row[2 * i + 1] = 1
                    mean_row[2 * j] = -mean_2
                    mean_row[2 * j + 1] = -1

                    # std difference row
                    std_row = [0] * num_params
                    std_row[2 * i] = std_1
                    std_row[2 * j] = -std_2

                    # Apply overlap weight (p_ij = s_ij)
                    mean_row = [
                        val * overlap_size * custom_mean_factor for val in mean_row
                    ]
                    std_row = [
                        val * overlap_size * custom_std_factor for val in std_row
                    ]

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
            mean_row = [0] * (2 * num_images)
            mean_row[2 * img_idx] = Mj
            mean_row[2 * img_idx + 1] = 1.0
            # we want: a_j*M_j + b_j - M_j = 0 => observed = M_j
            mean_obs = Mj

            # std constraint row
            std_row = [0] * (2 * num_images)
            std_row[2 * img_idx] = Vj
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
        print(
            f"Shape: constraint_matrix: {constraint_matrix.shape}, adjustment_params: {adjustment_params.shape}, observed_values_vector: {observed_values_vector.shape}"
        )
        print("constraint_matrix with labels:")
        # np.savetxt(sys.stdout, constraint_matrix, fmt="%16.3f")

        row_labels = []
        overlap_count = len(overlap_pairs)  # You must have recorded overlaps somewhere

        # Add two labels per overlap pair
        for i, j in overlap_pairs:
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

        print("adjustment_params:")
        np.savetxt(sys.stdout, adjustment_params, fmt="%18.3f")
        print("observed_values_vector:")
        np.savetxt(sys.stdout, observed_values_vector, fmt="%18.3f")

    print("-------------------- Apply adjustments and saving results")
    out_paths = []
    for img_idx in range(num_images):
        # for img_idx in [2]:
        adjusted_bands = []

        input_filename = os.path.basename(input_image_paths_array[img_idx])
        output_filename = (
            os.path.splitext(input_filename)[0] + output_global_basename + ".tif"
        )
        os.makedirs(os.path.join(output_image_folder, "Images"), exist_ok=True)
        output_path = os.path.join(output_image_folder, "Images", output_filename)
        out_paths.append(output_path)

        with rasterio.open(input_image_paths_array[img_idx]) as data:
            meta = data.meta.copy()
            meta.update({
                "driver": "GTiff",
                "count": num_bands,
                "nodata": all_nodata[img_idx],
            })

            with rasterio.open(output_path, "w", **meta) as data_out:
                for band_idx in range(num_bands):
                    a = all_adjustment_params[band_idx, 2 * img_idx, 0]
                    b = all_adjustment_params[band_idx, 2 * img_idx + 1, 0]

                    if tile_width_and_height_tuple:
                        windows = create_windows(data.width, data.height, tile_width_and_height_tuple[0], tile_width_and_height_tuple[1])
                    else:
                        windows = [Window(0, 0, data.width, data.height)]

                    for window in windows:
                        block = data.read(band_idx + 1, window=window)
                        mask = block != all_nodata[img_idx]
                        adjusted = np.where(mask, a * block + b, block)
                        data_out.write(adjusted, band_idx + 1, window=window)

        print(f"{img_idx} saved to: {output_path}")
    print("Global histogram matching done")
    return out_paths



def local_match(
    input_image_paths: List[str],
    output_image_folder: str,
    output_local_basename: str = "_local",
    custom_nodata_value: float = None,
    target_blocks_per_image: int = 100,
    alpha: float = 1.0,
    calculation_dtype_precision="float32",
    floor_value: Optional[float] = None,
    gamma_bounds: Optional[Tuple[float, float]] = None,
    output_dtype="float32",  # One of Byte, Int8, UInt16, Int16, UInt32, Int32, UInt64, Int64, Float32, Float64 as np strings
    projection: str = "EPSG:4326",
    debug_mode: bool = False,
    tile_width_and_height_tuple: tuple = None,
    correction_method: Literal["gamma", "linear"] = "gamma"
    ):
    """
    Performs local histogram matching on input raster images to align their intensity distributions
    to a reference distribution computed globally or locally. This process adjusts image brightness and
    contrast to make the distributions between the images comparable for consistency.

    Args:
    input_image_paths (List[str]): Paths to the input raster images to be processed.
    output_image_folder (str): Directory where the processed images and associated outputs will
    be saved.
    output_local_basename (str): Base name to append to the output image filenames.
    global_nodata_value (float, optional): Value representing no data for input raster images.
    Defaults to -9999.
    target_blocks_per_image (int, optional): Number of blocks to divide an image into for local
    histogram correction. Defaults to 100.
    alpha (float, optional): Scaling factor for histogram matching adjustments. Higher values
    lead to more significant adjustments. Defaults to 1.0.
    calculation_dtype_precision (str, optional): Data type used for calculations during histogram
    adjustments. Defaults to "float32".
    floor_value (Optional[float], optional): Minimum floor value applied during calculations.
    If None, the minimum is determined automatically. Defaults to None.
    gamma_bounds (Optional[Tuple[float, float]], optional): Lower and upper bounds for gamma
    correction values during intensity adjustment. Defaults to None.
    output_dtype (str, optional): Output data type for the processed raster images. One of Byte,
    Int8, UInt16, Int16, UInt32, Int32, UInt64, Int64, Float32, or Float64 based on numpy
    conventions. Defaults to "float32".
    projection (str, optional): Spatial reference system format for the output rasters.
    Defaults to "EPSG:4326".
    debug_mode (bool, optional): Enables debug outputs such as intermediate maps to assist
    troubleshooting. Defaults to False.

    Raises:
    ValueError: If `global_nodata_value` is not set and cannot be inferred from the first input image.
    RuntimeError: If an input raster file cannot be loaded correctly.

    """
    print("----------Starting Local Matching")

    print(f"Found {len(input_image_paths)} images")

    try:
        with rasterio.open(input_image_paths[0]) as ds: image_nodata_value = ds.nodata
    except:
        image_nodata_value = None

    if custom_nodata_value is None and image_nodata_value is None:
        print("custom_nodata_value not set and could not get one from the first band of the first image; using -9999")
        global_nodata_value = -9999

    if custom_nodata_value is None and image_nodata_value is not None:
        global_nodata_value = image_nodata_value

    if custom_nodata_value is not None:
        global_nodata_value = custom_nodata_value
        if image_nodata_value is not None and image_nodata_value != custom_nodata_value:
            print("Warning: image no data value has been overwritten by custom_nodata_value")


    print(f"Global nodata value: {global_nodata_value}")

    print("-------------------- Computing block size")
    # Its better to compute this offset right before gamma correciton, apply, then reverse
    # print('-------------------- Computing offset to make raster pixels > 0')
    # lowest_value: float = None
    # pixels_positive_offset: int = None
    # offset_image_paths = []
    #
    # # Find the lowest pixel value across all input rasters
    # for raster_path in input_image_paths:
    #     value = _get_lowest_pixel_value(raster_path)
    #     if lowest_value is None or value < lowest_value:
    #         lowest_value = value
    # print(f'Lowest_value: {lowest_value}')
    #
    # if lowest_value <= 0:
    #     pixels_positive_offset = int(abs(lowest_value))+1
    #     for raster_path in input_image_paths:
    #
    #         offset_image_path = os.path.join(output_image_folder,"OffsetImages",f"{os.path.splitext(os.path.basename(raster_path))[0]}_OffsetImage{os.path.splitext(raster_path)[1]}")
    #
    #         offset_image_paths.append(offset_image_path)
    #
    #         _add_value_to_raster(raster_path, offset_image_path, pixels_positive_offset)
    #         print(f"Offset of {pixels_positive_offset} saved: {offset_image_path}")
    # input_image_paths = offset_image_paths

    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder, exist_ok=True)

    # bounding rectangle
    bounding_rect = _get_bounding_rectangle(input_image_paths)
    print(f"Bounding rectangle: {bounding_rect}")

    M, N = _compute_block_size(
        input_image_paths, target_blocks_per_image, bounding_rect
    )
    print(f"Blocks(M,N): {M}, {N} = {M * N}")

    # -- Buffer boundary
    # print(f"Adjusted bounding rectangle (with 0.5-block offset): {bounding_rect}")
    # x_min, y_min, x_max, y_max = bounding_rect
    #
    # # compute block_width, block_height *before* expanding
    # block_width = (x_max - x_min) / N
    # block_height = (y_max - y_min) / M
    #
    # # Expand by half a block on each side
    # x_min -= block_width * 0.5
    # x_max += block_width * 0.5
    # y_min -= block_height * 0.5
    # y_max += block_height * 0.5
    #
    # # Re-assign the adjusted bounding_rect
    # bounding_rect = (x_min, y_min, x_max, y_max)

    # -- Alternatie aproach from the paper which I dont like
    # M, N = _compute_mosaic_coefficient_of_variation(input_image_paths, global_nodata_value)

    print("-------------------- Computing global reference block map")
    with rasterio.open(input_image_paths[0]) as ds: num_bands = ds.count

    block_reference_mean, block_reference_count = _compute_blocks(
        input_image_paths,
        bounding_rect,
        M,
        N,
        num_bands,
        nodata_value=global_nodata_value,
        tile_width_and_height_tuple=tile_width_and_height_tuple,
    )

    # block_reference_mean = _smooth_array(block_reference_mean, nodata_value=global_nodata_value)

    if debug_mode:
        _download_block_map(
            block_map=np.nan_to_num(block_reference_mean, nan=global_nodata_value),
            bounding_rect=bounding_rect,
            output_image_path=os.path.join(output_image_folder, 'BlockReferenceMean', "BlockReferenceMean.tif"),
            nodata_value=global_nodata_value,
            projection=projection,
        )

    out_paths = []
    for img_path in input_image_paths:
        print(f"-------------------- Processing: {img_path}")
        print(f"-------------------- Computing local block map")

        block_local_mean, block_local_count = _compute_blocks(
            [img_path],
            bounding_rect,
            M,
            N,
            num_bands,
            nodata_value=global_nodata_value,
            tile_width_and_height_tuple=tile_width_and_height_tuple,
        )

        # block_local_mean = _smooth_array(block_local_mean, nodata_value=global_nodata_value)

        out_name = (os.path.splitext(os.path.basename(img_path))[0] + output_local_basename)

        if debug_mode:
            _download_block_map(
                block_map=np.nan_to_num(block_local_count, nan=global_nodata_value),
                bounding_rect=bounding_rect,
                output_image_path=os.path.join(output_image_folder,"BlockLocalCount", out_name + "_BlockLocalCount" + '.tif'),
                nodata_value=global_nodata_value,
                projection=projection,
            )

        if debug_mode:
            _download_block_map(
                block_map=np.nan_to_num(block_local_mean, nan=global_nodata_value),
                bounding_rect=bounding_rect,
                output_image_path=os.path.join(output_image_folder, "BlockLocalMean", out_name + "_BlockLocalMean" + '.tif'),
                nodata_value=global_nodata_value,
                projection=projection,
            )

        print(f"-------------------- Computing local correction, applying, and saving")

        out_path = os.path.join(output_image_folder, 'Images', (out_name + '.tif'))
        if not os.path.exists(os.path.dirname(out_path)): os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with rasterio.open(img_path) as data_in:
            out_meta = data_in.meta.copy()
            out_meta.update({
                "count": num_bands,
                "dtype": output_dtype,
                "nodata": global_nodata_value
            })
            with rasterio.open(out_path, "w", **out_meta) as data_out:
                data_out.update_tags(nodata=global_nodata_value)
                if tile_width_and_height_tuple:
                    windows = create_windows(data_in.width, data_in.height, tile_width_and_height_tuple[0], tile_width_and_height_tuple[1])
                else:
                    windows = [Window(0, 0, data_in.width, data_in.height)]

                for window in windows:

                    for b in range(num_bands):
                        print(f"-------------------- For band {b + 1}")
                        win_transform = data_in.window_transform(window)

                        column_coords = win_transform[2] + np.arange(window.width) * win_transform[0]
                        row_coords = win_transform[5] + np.arange(window.height) * win_transform[4]

                        row_fractional_indices = np.clip(
                            ((bounding_rect[3] - row_coords) / (bounding_rect[3] - bounding_rect[1])) * M - 0.5,
                            0,
                            M - 1,
                        )
                        column_fractional_indices = np.clip(
                            ((column_coords - bounding_rect[0]) / (bounding_rect[2] - bounding_rect[0])) * N - 0.5,
                            0,
                            N - 1,
                        )

                        this_image_bounds = data_in.bounds
                        gc.collect()

                        arr_in = data_in.read(b + 1, window=window).astype(calculation_dtype_precision)
                        arr_out = np.full(arr_in.shape, global_nodata_value, dtype=calculation_dtype_precision)


                        valid_mask = arr_in != global_nodata_value

                        # if window is all nodata value, write and skip computation that require some data
                        if not np.any(valid_mask):
                            # All values are nodata; write the output and skip to next band
                            data_out.write(arr_out, b + 1, window=window)
                            del arr_out
                            gc.collect()
                            continue

                        valid_rows, valid_cols = np.where(valid_mask)

                        # Extract the band-specific local and reference maps
                        block_local_mean_band = block_local_mean[:, :, b]
                        block_reference_mean_band = block_reference_mean[:, :, b]

                        # if debug_mode:
                        #     _download_block_map(
                        #         block_map=np.where(valid_mask, 1, global_nodata_value),
                        #         bounding_rect=this_image_bounds,
                        #         output_image_path=os.path.join(output_image_folder, "ValidMasks", out_name + f"_ValidMask.tif"),
                        #         projection=projection,
                        #         nodata_value=global_nodata_value,
                        #         output_bands_map=(b+1,),
                        #         override_band_count=num_bands
                        #     )

                        # Ensure weighted interpolation handles only valid regions
                        reference_band = np.full(arr_in.shape, global_nodata_value, dtype=calculation_dtype_precision)
                        local_band = np.full(arr_in.shape, global_nodata_value, dtype=calculation_dtype_precision)

                        reference_band[valid_rows, valid_cols] = _weighted_bilinear_interpolation(
                            block_reference_mean_band,
                            # block_reference_mean[:, :, b],
                            column_fractional_indices[valid_cols],
                            row_fractional_indices[valid_rows],
                        )
                        del block_reference_mean_band
                        gc.collect()
                        local_band[valid_rows, valid_cols] = _weighted_bilinear_interpolation(
                            block_local_mean_band,
                            # block_local_count[:, :, b],
                            column_fractional_indices[valid_cols],
                            row_fractional_indices[valid_rows],
                        )

                        del column_fractional_indices, row_fractional_indices, block_local_mean_band
                        gc.collect()

                        # if debug_mode:
                        #     _download_block_map(
                        #         block_map=reference_band,
                        #         bounding_rect=this_image_bounds,
                        #         output_image_path=os.path.join(output_image_folder, "ReferenceBand", out_name + f"_ReferenceBand.tif"),
                        #         projection=projection,
                        #         nodata_value=global_nodata_value,
                        #         output_bands_map=(b+1,),
                        #         override_band_count=num_bands,
                        #     )
                        #
                        # if debug_mode:
                        #     _download_block_map(
                        #         block_map=local_band,
                        #         bounding_rect=this_image_bounds,
                        #         output_image_path=os.path.join(output_image_folder, "LocalBand", out_name + f"_LocalBand.tif"),
                        #         projection=projection,
                        #         nodata_value=global_nodata_value,
                        #         output_bands_map=(b+1,),
                        #         override_band_count=num_bands,
                        #     )

                        # valid_mask = valid_mask  # & (reference_band > 0) & (local_band > 0) # Mask if required but better to offset values <= 0

                        if correction_method == "gamma":
                            smallest_value = np.min([arr_in[valid_mask], reference_band[valid_mask], local_band[valid_mask]])
                            if smallest_value <= 0:
                                pixels_positive_offset = abs(smallest_value) + 1
                                arr_out[valid_mask], gammas = _apply_gamma_correction(
                                    arr_in[valid_mask] + pixels_positive_offset,
                                    reference_band[valid_mask] + pixels_positive_offset,
                                    local_band[valid_mask] + pixels_positive_offset,
                                    alpha,
                                )
                                arr_out[valid_mask] = arr_out[valid_mask] - pixels_positive_offset
                            else:
                                arr_out[valid_mask], gammas = _apply_gamma_correction(
                                    arr_in[valid_mask], reference_band[valid_mask], local_band[valid_mask], alpha
                                )
                            del gammas
                        elif correction_method == "linear":
                            arr_out[valid_mask] = arr_in[valid_mask] * (reference_band[valid_mask] / local_band[valid_mask])

                        del reference_band, local_band
                        gc.collect()

                        # if debug_mode:
                        #     gammas_array = np.full(arr_in.shape, global_nodata_value, dtype=calculation_dtype_precision)
                        #     gammas_array[valid_rows, valid_cols] = gammas
                        #     _download_block_map(
                        #         block_map=gammas_array,
                        #         bounding_rect=this_image_bounds,
                        #         output_image_path=os.path.join(output_image_folder, "Gamma", out_name + f"_Gamma.tif"),
                        #         projection=projection,
                        #         nodata_value=global_nodata_value,
                        #         output_bands_map=(b+1,),
                        #         override_band_count=num_bands,
                        #     )


                        data_out.write(arr_out, b + 1, window=window)
                        del arr_out
                        gc.collect()

                out_paths.append(out_path)
                print(f"Saved: {out_path}")

    print("Local histogram matching done")
    return out_paths
