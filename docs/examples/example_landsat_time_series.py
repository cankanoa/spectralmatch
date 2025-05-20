# %% Landsat Time Series
# This notebook demonstrates how to preprocess Landsat 8-9 into a time series with spectralmatch.
# Starting from 5 Landsat 8-9 OLI/TIRS C2 L1 images, the process includes clipping clouds with OmniCloudMask, masking high NDVI areas as Pseudo Invariant Features (PIFs), applying global regression Relative Radiometric Normalization, fine-tuning overlap areas with local block adjustment, and before vs after statistics.
# This script is set up to perform matching on all tif files from a folder within the working directory called "Input" e.g. working_directory/Input/*.tif.

# %% Setup
import os
from spectralmatch import *

# Important: If this does not automatically find the correct CWD, manually copy the path to the /data_worldview3 folder
working_directory = os.path.join(os.getcwd(), "data_landsat")
print(working_directory)

input_folder = os.path.join(working_directory, "Input")
global_folder = os.path.join(working_directory, "GlobalMatch")
local_folder = os.path.join(working_directory, "LocalMatch")

mask_cloud_folder = os.path.join(working_directory, "MaskCloud")
mask_vegetation_folder = os.path.join(working_directory, "MaskVegetation")

masked_folder = os.path.join(working_directory, "Masked")

window_size = 128
num_workers = 5

# %% Create cloud masks
input_image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(".tif")]

for path in input_image_paths:
    create_cloud_mask_with_omnicloudmask(
        path,
        5,
        3,
        8,
        os.path.join(mask_cloud_folder, f"{os.path.splitext(os.path.basename(path))[0]}_CloudMask.tif"),
        # down_sample_m=10
    )

input_mask_rasters_paths = [os.path.join(mask_cloud_folder, f) for f in os.listdir(mask_cloud_folder) if f.lower().endswith(".tif")]

for path in input_mask_rasters_paths:
    post_process_raster_cloud_mask_to_vector(
        path,
        os.path.join(mask_cloud_folder, f"{os.path.splitext(os.path.basename(path))[0]}.gpkg"),
        None,
        {1: 50},
        {0: 0, 1: 1, 2: 1, 3: 1}
    )

# %% Use cloud masks
input_image_paths = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(".tif")])
input_mask_vectors = sorted([os.path.join(mask_cloud_folder, f) for f in os.listdir(mask_cloud_folder) if f.lower().endswith(".gpkg")])
output_paths = sorted([os.path.join(masked_folder, os.path.splitext(os.path.basename(path))[0] + "_CloudMasked.tif") for path in input_image_paths])

for input_path, vector_path, output_path in zip(input_image_paths, input_mask_vectors, output_paths):
    mask_image_with_vector(
        input_path,
        vector_path,
        output_path,
        {"value": 0},
        True
    )

# %% Mask trees as a non-PIF for isolated analysis
input_image_paths = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(".tif")])
raster_mask_paths = sorted([os.path.join(mask_vegetation_folder, os.path.splitext(os.path.basename(path))[0] + "_VegetationMask.tif") for path in input_image_paths])
vectors_mask_paths = sorted([os.path.join(mask_vegetation_folder, os.path.splitext(os.path.basename(path))[0] + ".gpkg") for path in input_image_paths])
merged_vector_pif_path = working_directory + "/Pifs.gpkg"


for input_path, raster_path in zip(input_image_paths, raster_mask_paths):
    create_ndvi_mask(
        input_path,
        raster_path,
        5,
        4,
    )

for raster_path, vector_path in zip(raster_mask_paths, vectors_mask_paths):
    post_process_threshold_to_vector(
        raster_path,
        vector_path,
        0.2,
        "<=",
    )

merge_vectors(
    vectors_mask_paths,
    merged_vector_pif_path,
    method="intersection"
    )

# %% Global matching
vector_mask_path = working_directory + "Pifs.gpkg"

global_regression(
    masked_folder,
    (global_folder, '_GlobalMatch'),
    custom_mean_factor = 3, # Defualt 1; 3 often works better to 'move' the spectral mean of images closer together
    vector_mask_path=("exclude",vector_mask_path),
    debug_logs=True,
    window_size=window_size,
    parallel_workers=num_workers,
    )


# %% Local matching
input_image_paths = [os.path.join(global_folder, f) for f in os.listdir(global_folder) if f.lower().endswith(".tif")]
local_folder = os.path.join(working_directory, "LocalMatch")
global_image_paths_array = [os.path.join(global_folder, f) for f in os.listdir(global_folder) if f.lower().endswith(".tif")]

matched_local_images_paths = local_block_adjustment(
    global_image_paths_array,
    local_folder,
    number_of_blocks=100,
    debug_logs=True,
    window_size=window_size,
    parallel=num_workers,
    )

merge_rasters(
    matched_local_images_paths, # Rasters are layered with the last ones on top
    os.path.join(working_directory, "MatchedLocalImages.tif"),
    window_size=window_size,
    )

print("Done with global and local histogram matching")


# %% Statistics
