import os

from spectralmatch.voronoi_centerline_seamline import voronoi_centerline_seamline

working_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_data")
input_folder = os.path.join(working_directory, "Output", "LocalMatch", "Images")
input_image_paths_array = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(".tif")]

voronoi_centerline_seamline(
    input_image_paths_array,
    './example_data/Output/seamline_mask.gpkg')