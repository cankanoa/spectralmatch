import itertools

from osgeo import gdal
import numpy as np
import rasterio
import matplotlib.pyplot as plt


def compare_image_spectral_profiles(
    input_image_dict: dict,
    output_figure_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
    ):
    """
    Compares spectral profiles of multiple images by plotting median and interquartile ranges.

    Args:
        input_image_dict (dict): Mapping of labels to image file paths.
        output_figure_path (str): Path to save the output plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.

    Outputs:
        Saves a spectral profile comparison figure to the specified path.
    """

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


def compare_image_spectral_profiles_pairs(
    image_groups_dict: dict,
    output_figure_path: str,
    ):
    """
    Plots paired spectral profiles for before-and-after image comparisons.

    Args:
        image_groups_dict (dict): Mapping of labels to image path pairs (before, after).
        output_figure_path (str): Path to save the resulting comparison figure.

    Outputs:
        Saves a spectral comparison plot showing pre- and post-processing profiles.
    """

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


def compare_spatial_spectral_difference_average(
    input_overlapping_image_pair_path: list,
    output_image_path: str,
    ):
    """
    Generates a heatmap of the average spectral difference between two overlapping images.

    Args:
        input_overlapping_image_pair_path (list): List containing exactly two image paths (pre and post).
        output_image_path (str): Path to save the resulting difference visualization.

    Outputs:
        Saves a heatmap image illustrating spatial spectral differences.

    Raises:
        ValueError: If the list does not contain exactly two images or if image dimensions differ.
    """

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


def compare_spatial_spectral_difference_individual_bands(
    input_overlapping_image_pair_paths: tuple,
    output_image_path: str,
    ):
    """
    Creates a color-coded visualization of spectral differences per band between two overlapping images.

    Args:
        input_overlapping_image_pair_paths (tuple): Tuple of two image paths (before, after).
        output_image_path (str): Path to save the RGB difference visualization as a PNG.

    Outputs:
        Saves a PNG image where color represents the dominant band of spectral difference and brightness indicates magnitude.

    Raises:
        ValueError: If input images differ in shape.
    """

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

    diff = np.abs(img1 - img2).astype(np.float32)  # (bands, height, width)

    global_min = diff.min()  # often 0
    global_max = diff.max()

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

    band_colors = [default_colors[i % len(default_colors)] for i in range(num_bands)]
    band_colors = np.array(band_colors, dtype=np.float32)  # shape: (num_bands, 3)

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


    brightness = (sum_diff - global_min) / (global_max - global_min + 1e-10)
    brightness = np.clip(brightness, 0.0, 1.0)
    # Multiply the RGB by brightness
    output_rgb *= brightness.reshape(1, height, width)

    output_rgb_for_plot = np.transpose(output_rgb, (1, 2, 0))

    # Ensure the data is in 0..1
    output_rgb_for_plot = np.clip(output_rgb_for_plot, 0, 1)

    # Use plt.imsave to write out a PNG
    plt.imsave(output_image_path, output_rgb_for_plot)

    print(f"Saved difference visualization PNG to: {output_image_path}")