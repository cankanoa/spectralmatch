import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os
import itertools
import matplotlib.colors as mcolors


import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

def compare_image_spectral_profiles(
        title,
        xlabel,
        ylabel,
        input_image_dict,
        output_figure_path,
        ):
    plt.figure(figsize=(10, 6))
    colors = itertools.cycle(plt.cm.tab10.colors)  # Cycle through colors
    spectral_profiles = []
    labels = []

    for label, image_path in input_image_dict.items():
        dataset = gdal.Open(image_path)
        if dataset is None:
            print(f"Failed to open {image_path}")
            continue

        image_data = dataset.ReadAsArray()
        if image_data.ndim == 3:
            bands, height, width = image_data.shape
        else:
            bands, height, width = 1, *image_data.shape
            image_data = np.expand_dims(image_data, axis=0)

        image_data = image_data.reshape(bands, -1)
        mean_spectral = np.median(image_data, axis=1)
        q25, q75 = np.percentile(image_data, [25, 75], axis=1)
        spectral_profiles.append((mean_spectral, q25, q75))
        labels.append(label)

    for i, (mean_spectral, q25, q75) in enumerate(spectral_profiles):
        color = next(colors)  # Assign unique color
        plt.plot(range(1, len(mean_spectral) + 1), mean_spectral, color=color, label=labels[i])
        plt.fill_between(range(1, len(mean_spectral) + 1), q25, q75, color=color, alpha=0.3)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_figure_path, dpi=300)
    plt.close()
    print(f"Figure saved to: {output_figure_path}")

# compare_image_spectral_profiles(
#     "Digital Number Spectral Profile Comparison",
#     'Band',
#     'Digital Number(0-2,047)',
#     {
#         'Image A': '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/DN/17DEC08211758-M1BS-016445319010_01_P003_Graph_Entire.tif',
#         'Image B': '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/DN/17DEC08211841-M1BS-016445318010_01_P016_Graph_Entire.tif'
#     },
#      "/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/DN/SpectralComparison_DN.png",
#      )
#
# compare_image_spectral_profiles(
#     "Reflectance Spectral Profile Comparison",
#     'Band',
#     'Reflectance(0-10,000)',
#     {
#         'Image A': '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/FLAASH_OrthoFromUSGSLidar_Pansharp/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFromUSGSLidar_Pansharp_Graph_Entire.tif',
#         'Image B': '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/FLAASH_OrthoFromUSGSLidar_Pansharp/17DEC08211841-M1BS-016445318010_01_P016_FLAASH_OrthoFromUSGALidar_Pansharp_Graph_Entire.tif'
#     },
#     "/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/FLAASH_OrthoFromUSGSLidar_Pansharp/SpectralComparison_FLAASH.png",
#      )
#
# compare_image_spectral_profiles(
#         "Globally Matched Reflectance Spectral Profile Comparison",
#         'Band',
#         'Reflectance(0-10,000)',
#         {
#             'Image A': '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/FLAASH_OrthoFromUSGSLidar_Pansharp_GlobalMatch/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFromUSGSLidar_Pansharp_GlobalMatch_Graph_Entire.tif',
#             'Image B': '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/FLAASH_OrthoFromUSGSLidar_Pansharp_GlobalMatch/17DEC08211841-M1BS-016445318010_01_P016_FLAASH_OrthoFromUSGALidar_Pansharp_GlobalMatch_Graph_Entire.tif'
#         },
#         "/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/FLAASH_OrthoFromUSGSLidar_Pansharp_GlobalMatch/SpectralComparison_FLAASH_OrthoFromUSGSLidar_Pansharp_GlobalMatch.png",
#          )
#
# compare_image_spectral_profiles(
#         "Globally and Locally Matched Reflectance Spectral Profile Comparison",
#         'Band',
#         'Reflectance(0-10,000)',
#         {
#             'Image A': '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/FLAASH_OrthoFromUSGSLidar_Pansharp_GlobalMatch_LocalMatch/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFromUSGSLidar_Pansharp_GlobalMatch_LocalMatch_Graph_Entire.tif',
#             'Image B': '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/FLAASH_OrthoFromUSGSLidar_Pansharp_GlobalMatch_LocalMatch/17DEC08211841-M1BS-016445318010_01_P016_FLAASH_OrthoFromUSGALidar_Pansharp_GlobalMatch_LocalMatch_Graph_Entire.tif'
#         },
#         "/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/FLAASH_OrthoFromUSGSLidar_Pansharp_GlobalMatch_LocalMatch/SpectralComparison_FLAASH_OrthoFromUSGSLidar_Pansharp_GlobalMatch_LocalMatch.png",
#          )


def compare_image_spectral_profiles_pairs(image_groups_dict, output_figure_path):
    plt.figure(figsize=(10, 6))
    colors = itertools.cycle(plt.cm.tab10.colors)  # Cycle through colors

    for label, group in image_groups_dict.items():
        if len(group) == 2:  # Ensure paired comparison
            image_path1, image_path2 = group
            color = next(colors)  # Assign the same color to both images

            for i, image_path in enumerate([image_path1, image_path2]):
                with rasterio.open(image_path) as src:
                    img = src.read()
                    num_bands = img.shape[0]
                    img_reshaped = img.reshape(num_bands, -1)
                    nodata_value = src.nodata
                    if nodata_value is not None:
                        img_reshaped = np.where(img_reshaped == nodata_value, np.nan, img_reshaped)
                    mean_spectral = np.nanmean(img_reshaped, axis=1)
                    bands = np.arange(1, num_bands + 1)
                    linestyle = 'dashed' if i == 0 else 'solid'
                    plt.plot(bands, mean_spectral, linestyle=linestyle, color=color, label=f"{label} - {'Before' if i == 0 else 'After'}")

    plt.xlabel("Band Number")
    plt.ylabel("Reflectance(0-10,000)")
    plt.title("Pre and Post Spectral Match Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_figure_path, dpi=300)
    plt.close()
    print(f"Figure saved to: {output_figure_path}")

input_image_groups = {
    'Image A': [
        '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/FLAASH_OrthoFromUSGSLidar_Pansharp/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFromUSGSLidar_Pansharp_Graph_Entire.tif',
        '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/FLAASH_OrthoFromUSGSLidar_Pansharp_GlobalMatch_LocalMatch/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFromUSGSLidar_Pansharp_GlobalMatch_LocalMatch_Graph_Entire.tif'
    ],
    'Image B': [
        '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/FLAASH_OrthoFromUSGSLidar_Pansharp/17DEC08211841-M1BS-016445318010_01_P016_FLAASH_OrthoFromUSGALidar_Pansharp_Graph_Entire.tif',
        '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/FLAASH_OrthoFromUSGSLidar_Pansharp_GlobalMatch_LocalMatch/17DEC08211841-M1BS-016445318010_01_P016_FLAASH_OrthoFromUSGALidar_Pansharp_GlobalMatch_LocalMatch_Graph_Entire.tif'
    ]
    }

compare_image_spectral_profiles_pairs(input_image_groups, '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/SpectralProfileCompare.png')

# def compare_image_spectral_profiles_pairs_with_iqr(image_groups_dict, output_figure_path):
#     plt.figure(figsize=(10, 6))
#     colors = itertools.cycle(plt.cm.tab10.colors)  # Cycle through colors
#
#     for label, group in image_groups_dict.items():
#         if len(group) == 2:  # Ensure paired comparison
#             image_path1, image_path2 = group
#             color = next(colors)  # Assign the same color to both images
#             darker_color = mcolors.to_rgba(color, alpha=0.5)  # Slightly darker shade
#
#             for i, image_path in enumerate([image_path1, image_path2]):
#                 with rasterio.open(image_path) as src:
#                     img = src.read()
#                     num_bands = img.shape[0]
#                     img_reshaped = img.reshape(num_bands, -1)
#                     nodata_value = src.nodata
#                     if nodata_value is not None:
#                         img_reshaped = np.where(img_reshaped == nodata_value, np.nan, img_reshaped)
#                     mean_spectral = np.nanmean(img_reshaped, axis=1)
#                     q25, q75 = np.nanpercentile(img_reshaped, [25, 75], axis=1)
#                     bands = np.arange(1, num_bands + 1)
#                     linestyle = 'dashed' if i == 0 else 'solid'
#                     edge_color = darker_color  # Use darker color for border
#                     plt.plot(bands, mean_spectral, linestyle=linestyle, color=color, label=f"{label} - {'Before' if i == 0 else 'After'}")
#                     plt.fill_between(bands, q25, q75, facecolor=color, alpha=0.2, edgecolor=edge_color, linewidth=1.5, linestyle='dotted' if i == 0 else 'solid')
#
#     plt.xlabel("Band Number")
#     plt.ylabel("Reflectance")
#     plt.title("Spectral Profile Comparisons")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(output_figure_path, dpi=300)
#     plt.close()
#     print(f"Figure saved to: {output_figure_path}")
#
# compare_image_spectral_profiles_pairs_with_iqr(input_image_groups, '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/SpectralProfileCompare_IQR.png')


def compare_spatial_spectral_difference_average(input_overlapping_image_pair_path, output_image_path):
    if len(input_overlapping_image_pair_path) != 2:
        raise ValueError("Function requires exactly two image paths for comparison.")

    image_path1, image_path2 = input_overlapping_image_pair_path

    with rasterio.open(image_path1) as src1, rasterio.open(image_path2) as src2:
        img1 = src1.read()  # Read all bands (shape: bands, height, width)
        img2 = src2.read()

        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions for comparison.")

        # Compute absolute spectral difference per band
        diff = np.abs(img2 - img1)
        mean_diff = np.mean(diff, axis=0)  # Average across bands for visualization

        plt.figure(figsize=(10, 6))
        plt.imshow(mean_diff, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label="Spectral Difference")
        plt.title("Spatial Spectral Difference (Post - Pre)")
        plt.axis("off")

        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Spatial spectral difference figure saved to: {output_image_path}")

# compare_spatial_spectral_difference_average(
#     ['/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/FLAASH_OrthoFromUSGSLidar_Pansharp/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFromUSGSLidar_Pansharp_Graph_Entire.tif',
#      '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/FLAASH_OrthoFromUSGSLidar_Pansharp_GlobalMatch/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFromUSGSLidar_Pansharp_GlobalMatch_Graph_Entire.tif'
#      ],
#     '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/SpatialSpectralDifferenceCompare.png'
#     )

import numpy as np
import rasterio
import matplotlib.pyplot as plt

def compare_spatial_spectral_difference_individual_bands(
        input_overlapping_image_pair_paths,
        output_image_path
):
    """
Compare two overlapping images on a per-band, per-pixel basis and produce
a color-coded difference visualization as a PNG image.

Parameters
----------
input_overlapping_image_pair_paths : tuple or list of str
A tuple/list of exactly two file paths to the images to compare.
output_image_path : str
The file path (e.g. "output.png") where the resulting visualization will be saved as a PNG.
    """
    # -------------------------------------------------------------------------
    # 1. Read the input images as NumPy arrays
    # -------------------------------------------------------------------------
    path1, path2 = input_overlapping_image_pair_paths
    with rasterio.open(path1) as src1, rasterio.open(path2) as src2:
        img1 = src1.read()  # shape: (num_bands, height, width)
        img2 = src2.read()  # shape: (num_bands, height, width)

        if img1.shape != img2.shape:
            raise ValueError(
                f"Input images do not have the same shape:\n"
                f" Image1 shape: {img1.shape}, Image2 shape: {img2.shape}"
            )

        num_bands, height, width = img1.shape

    # -------------------------------------------------------------------------
    # 2. Compute absolute difference per band, per pixel
    # -------------------------------------------------------------------------
    diff = np.abs(img1 - img2).astype(np.float32)  # (bands, height, width)

    # -------------------------------------------------------------------------
    # 3. Compute global min/max difference for optional brightness scaling
    # -------------------------------------------------------------------------
    global_min = diff.min()  # often 0
    global_max = diff.max()

    # -------------------------------------------------------------------------
    # 4. Assign colors for each band
    # -------------------------------------------------------------------------
    # You can assign your own color palette. Below is a simple example:
    default_colors = [
        (1.0, 0.0, 0.0),  # Band 1 -> Red
        (0.0, 1.0, 0.0),  # Band 2 -> Green
        (0.0, 0.0, 1.0),  # Band 3 -> Blue
        (1.0, 1.0, 0.0),  # Band 4 -> Yellow
        (1.0, 0.0, 1.0),  # Band 5 -> Magenta
        (0.0, 1.0, 1.0),  # Band 6 -> Cyan
        (1.0, 0.5, 0.0),  # Band 7 -> Orange
        # etc. Add more if needed
    ]
    # If there are more bands than colors, cycle through them
    band_colors = [default_colors[i % len(default_colors)] for i in range(num_bands)]
    band_colors = np.array(band_colors, dtype=np.float32)  # shape: (num_bands, 3)

    # -------------------------------------------------------------------------
    # 5. Create the output visualization (RGB image) by blending band colors
    #    according to each band’s fraction of the total difference.
    # -------------------------------------------------------------------------
    # Initialize an empty float array for RGB: shape = (3, height, width)
    output_rgb = np.zeros((3, height, width), dtype=np.float32)

    # sum of differences per pixel across all bands -> shape: (height, width)
    sum_diff = diff.sum(axis=0)
    sum_diff_safe = np.where(sum_diff == 0, 1e-10, sum_diff)  # avoid division-by-zero

    # fraction_diff[b, y, x] = diff[b, y, x] / sum_diff[y, x]
    fraction_diff = diff / sum_diff_safe

    # Accumulate weighted colors
    for b in range(num_bands):
        # band_colors[b] is shape (3,)
        # fraction_diff[b] is shape (height, width)
        output_rgb += fraction_diff[b] * band_colors[b].reshape(3, 1, 1)

    # -------------------------------------------------------------------------
    # 6. Optionally scale the brightness by overall difference magnitude
    #    so pixels with greater differences appear brighter.
    # -------------------------------------------------------------------------
    brightness = (sum_diff - global_min) / (global_max - global_min + 1e-10)
    brightness = np.clip(brightness, 0.0, 1.0)
    # Multiply the RGB by brightness
    output_rgb *= brightness.reshape(1, height, width)

    # -------------------------------------------------------------------------
    # 7. Convert to the format expected by matplotlib (height x width x 3)
    #    and save as a PNG with matplotlib.
    # -------------------------------------------------------------------------
    # Currently: output_rgb has shape (3, height, width).
    # We transpose to (height, width, 3).
    output_rgb_for_plot = np.transpose(output_rgb, (1, 2, 0))

    # Ensure the data is in 0..1
    output_rgb_for_plot = np.clip(output_rgb_for_plot, 0, 1)

    # Use plt.imsave to write out a PNG
    plt.imsave(output_image_path, output_rgb_for_plot)

    print(f"Saved difference visualization PNG to: {output_image_path}")



# compare_spatial_spectral_difference_individual_bands(
#     ('/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/FLAASH_OrthoFromUSGSLidar_Pansharp/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFromUSGSLidar_Pansharp_Graph_Entire.tif',
#      '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/FLAASH_OrthoFromUSGSLidar_Pansharp_GlobalMatch/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFromUSGSLidar_Pansharp_GlobalMatch_Graph_Entire.tif'
#      ),
#     '/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/GraphSite/Images/SpatialSpectralDifferenceCompare_IndividualBands.png'
# )


