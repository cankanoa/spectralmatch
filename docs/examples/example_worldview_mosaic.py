# %% Worldview Mosaic
# This file demonstrates how to preprocess Worldview3 imagery into a mosaic. Starting from two overlapping Worldview3 images in reflectance, the process includes global matching, local matching, starting from saved block maps (optional for demonstration purposes), generating seamlines, and marging images, and and before vs after statistics.
# This script is setup to perform matching on all tif files from a folder within the working directory called "Input" e.g. working_directory/Input/*.tif.

# %% Setup
import os
import importlib

from spectralmatch.match.global_regression import global_regression
from spectralmatch.match.local_block_adjustment import local_block_adjustment
from spectralmatch.handlers import merge_rasters, mask_rasters
from spectralmatch.voronoi_center_seamline import voronoi_center_seamline

working_directory = os.path.join(os.getcwd(), "data_worldview3")
# This script is setup to perform matching on all tif files from a folder within the working directory called "input" e.g. working_directory/input/*.tif.

vector_mask_path = working_directory + "/Input/Masks.gpkg"

input_folder = os.path.join(working_directory, "Input")
global_folder = os.path.join(working_directory, "GlobalMatch")
local_folder = os.path.join(working_directory, "LocalMatch")


# %% Global matching
input_image_paths_array = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(".tif")]

matched_global_images_paths = global_regression(
    input_image_paths_array,
    global_folder,
    custom_mean_factor = 3, # Defualt 1; 3 often works better to 'move' the spectral mean of images closer together
    custom_std_factor = 1,
    # vector_mask_path=vector_mask_path,
    debug_logs=True,
    window_size=128,
    parallel_workers=4
    )


# %% Local matching
global_image_paths_array = [os.path.join(global_folder, f) for f in os.listdir(global_folder) if f.lower().endswith(".tif")]

matched_local_images_paths = local_block_adjustment(
    global_image_paths_array,
    local_folder,
    number_of_blocks=100,
    debug_logs=True,
    window_size=128,
    parallel_workers="cpu",
    )


# %% Start from saved block maps
saved_reference_path = os.path.join(local_folder, "BlockReferenceMean", "BlockReferenceMean.tif")
saved_local_folder_path = os.path.join(local_folder, "BlockLocalMean")
saved_local_paths = [os.path.join(saved_local_folder_path, f) for f in os.listdir(saved_local_folder_path) if f.lower().endswith(".tif")]

new_local_folder = os.path.join(working_directory, "New_LocalMatch")

matched_local_images_paths = local_block_adjustment(
    global_image_paths_array,
    new_local_folder,
    number_of_blocks=100,
    debug_logs=True,
    window_size=512,
    parallel_workers="cpu",
    pre_computed_block_map_paths=(saved_reference_path, saved_local_paths)
    )


# %% Generate seamlines
input_image_paths_array = [os.path.join(local_folder, f) for f in os.listdir(local_folder) if f.lower().endswith(".tif")]
output_vector_mask = os.path.join(working_directory, "ImageMasks.gpkg")

voronoi_center_seamline(
    input_image_paths_array,
    output_vector_mask,
    )


# %% Mask and merge
input_image_paths_array = sorted([os.path.join(local_folder, f) for f in os.listdir(local_folder) if f.lower().endswith(".tif")])
output_folder = os.path.join(working_directory, "MaskedImages")
masked_image_paths = sorted([os.path.join(output_folder, os.path.splitext(os.path.basename(path))[0] + "_MaskedImages.tif") for path in input_image_paths_array])
input_vector_mask_path = os.path.join(working_directory, "ImageMasks.gpkg")
output_merged_image_path = os.path.join(working_directory, "MergedImage.tif")

mask_rasters(
    input_image_paths_array,
    masked_image_paths,
    input_vector_mask_path,
    tap=True,
    debug_logs=True,
    split_mask_by_attribute="image",
    window_size=100,
    )

merge_rasters(
    masked_image_paths,
    output_merged_image_path,
    window_size=100,
    debug_logs=True,
)

# %% Statistics
# To visually see the difference make sure to merge input images so that their histograms match