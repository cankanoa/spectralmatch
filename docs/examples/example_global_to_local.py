import os

from spectralmatch import global_regression, local_block_adjustment
from spectralmatch import merge_rasters

# -------------------- Parameters
working_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_data")
# This script is setup to perform matching on all tif files from a folder within the working directory called "input" e.g. working_directory/input/*.tif.

vector_mask_path = working_directory + "/Input/Masks.gpkg"

input_folder = os.path.join(working_directory, "Input")
global_folder = os.path.join(working_directory, "GlobalMatch")
local_folder = os.path.join(working_directory, "LocalMatch")

# -------------------- Global histogram matching
input_image_paths_array = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(".tif")]

matched_global_images_paths = global_regression(
    input_image_paths_array,
    global_folder,
    custom_mean_factor = 3, # Defualt 1; 3 often works better to 'move' the spectral mean of images closer together
    custom_std_factor = 1,
    # vector_mask_path=vector_mask_path,
    debug_mode=False,
    window_size=(512, 512),
    parallel=True,
    )

merge_rasters(
    matched_global_images_paths, # Rasters are layered with the last ones on top
    os.path.join(working_directory, "MatchedGlobalImages.tif"),
    tile_width_and_height_tuple=(512, 512),
    )

# -------------------- Local histogram matching
global_image_paths_array = [os.path.join(global_folder, f) for f in os.listdir(global_folder) if f.lower().endswith(".tif")]

matched_local_images_paths = local_block_adjustment(
    global_image_paths_array,
    local_folder,
    target_blocks_per_image=100,
    debug_mode=True,
    window_size="block",
    parallel=True,
    )

merge_rasters(
    matched_local_images_paths, # Rasters are layered with the last ones on top
    os.path.join(working_directory, "MatchedLocalImages.tif"),
    tile_width_and_height_tuple=(512, 512),
    )

print("Done with global and local histogram matching")

# Statistics
# To visually see the difference make sure to merge input images so that their histograms match