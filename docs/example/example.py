import os

from spectralmatch.process import global_match, local_match
from spectralmatch.handlers import merge_rasters

# -------------------- Parameters
working_directory = os.path.dirname(os.path.abspath(__file__))
# This script is setup to perform matching on all tif files from a folder within the working directory called "input" e.g. working_directory/input/*.tif.

vector_mask_path = working_directory + "/input/Masks.gpkg"

input_folder = os.path.join(working_directory, "input")
global_folder = os.path.join(working_directory, "output/global_match")
local_folder = os.path.join(working_directory, "output/local_match")

# -------------------- Global histogram matching
input_image_paths_array = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(".tif")]

matched_global_images_paths = global_match(
    input_image_paths_array,
    global_folder,
    custom_mean_factor = 3, # Defualt 1; 3 often works better to 'move' the spectral mean of images closer together
    custom_std_factor = 1,
    # vector_mask_path=vector_mask_path,
    debug_mode=True,
    tile_width_and_height_tuple=(512, 512),
    )

merge_rasters(
    matched_global_images_paths, # Rasters are layered with the last ones on top
    os.path.join(working_directory, "output/global_match/matched_global_images.tif"),
    )

# -------------------- Local histogram matching
global_image_paths_array = [os.path.join(f"{global_folder}/images", f) for f in os.listdir(f"{global_folder}/images") if f.lower().endswith(".tif")]

matched_local_images_paths = local_match(
    global_image_paths_array,
    local_folder,
    target_blocks_per_image=100,
    projection="EPSG:6635",
    debug_mode=True,
    custom_nodata_value=-9999,
    tile_width_and_height_tuple=(512, 512),
    )

merge_rasters(
    matched_local_images_paths, # Rasters are layered with the last ones on top  
    os.path.join(working_directory, "output/local_match/matched_local_images.tif"),
    )

print("Done with global and local histogram matching")