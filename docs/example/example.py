import os

from spectralmatch.process import global_match, local_match
script_dir = os.path.dirname(os.path.abspath(__file__))

# -------------------- Global params
input_folder = os.path.join(script_dir, "input")
global_folder = os.path.join(script_dir, "output/global_match")  # This is the output of global match
custom_mean_factor = 3  # Defualt 1; 3 often works better to 'move' the spectral mean of images closer together
custom_std_factor = 1  # Defualt 1

# -------------------- Local params
local_folder = os.path.join(script_dir, "output/local_match")


# -------------------- Global Histogram Match Mulispectral Images
input_image_paths_array = [
    os.path.join(input_folder, f)
    for f in os.listdir(input_folder)
    if f.lower().endswith(".tif")
]

global_match(
    input_image_paths_array,
    global_folder,
    custom_mean_factor,
    custom_std_factor,
)

# -------------------- Local Histogram Match Mulispectral Images
global_image_paths_array = [
    os.path.join(f"{global_folder}/images", f)
    for f in os.listdir(f"{global_folder}/images")
    if f.lower().endswith(".tif")
]

local_match(
    global_image_paths_array,
    local_folder,
    target_blocks_per_image=100,
    projection="EPSG:6635",
    debug_mode=True,
    global_nodata_value=-32768,
)

print("Done with global and local histogram matching")