from local_histogram_match import local_histogram_match
from global_histogram_match import global_histogram_match
import os

def run_automated_image_mosaicing():
    # Spectral matching from DOI: 10.1016/j.isprsjprs.2017.08.002

    # -------------------- Global Histogram Match Mulispectral Images
    input_folder = '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/Mosaic/PuuWaawaa_20171208/Source'
    global_folder = '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/Mosaic/PuuWaawaa_20171208/GlobalMatch'
    output_global_basename = "_GlobalMatch"
    custom_mean_factor = 3 # Defualt 1; 3 works well sometimes
    custom_std_factor = 1 # Defualt 1

    input_image_paths_array = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith('.tif')]
    # global_histogram_match(
    #     input_image_paths_array,
    #     global_folder,
    #     output_global_basename,
    #     custom_mean_factor,
    #     custom_std_factor
    #     )

    # -------------------- Local Histogram Match Mulispectral Images
    input_image_paths_array = [os.path.join(f'{global_folder}/images', f) for f in os.listdir(f'{global_folder}/images') if f.lower().endswith('.tif')]
    local_folder = '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/Mosaic/PuuWaawaa_20171208/LocalMatch'
    output_local_basename = "_LocalMatch"

    local_histogram_match(
        input_image_paths_array,
        local_folder,
        output_local_basename,
        target_blocks_per_image = 100,
        projection="EPSG:6635",
        debug_mode=True,
        global_nodata_value=-32768
        )

    print('Done with main match imagery')

run_automated_image_mosaicing()