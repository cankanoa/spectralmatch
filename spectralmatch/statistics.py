import itertools
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt


def compare_image_spectral_profiles(
    input_image_dict,
    output_figure_path,
    title,
    xlabel,
    ylabel,
):
    """
    Compares spectral profiles of multiple images by plotting median and interquartile ranges.

    Args:
        input_image_dict (dict): Mapping of labels to image file paths:
            {
            'Image A': '/image/a.tif',
            'Image B': '/image/b.tif'
            }
        output_figure_path (str): Path to save the output plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.

    Outputs:
        Saves a spectral profile comparison figure to the specified path.
    """
    os.makedirs(os.path.dirname(output_figure_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    colors = itertools.cycle(plt.cm.tab10.colors)
    spectral_profiles = []
    labels = []

    for label, image_path in input_image_dict.items():
        try:
            with rasterio.open(image_path) as src:
                image_data = src.read()  # shape: (bands, height, width)
        except Exception as e:
            print(f"Failed to open {image_path}: {e}")
            continue

        bands, height, width = image_data.shape
        reshaped = image_data.reshape(bands, -1)
        median = np.median(reshaped, axis=1)
        q25, q75 = np.percentile(reshaped, [25, 75], axis=1)
        spectral_profiles.append((median, q25, q75))
        labels.append(label)

    for i, (median, q25, q75) in enumerate(spectral_profiles):
        color = next(colors)
        x = range(1, len(median) + 1)
        plt.plot(x, median, color=color, label=labels[i])
        plt.fill_between(x, q25, q75, color=color, alpha=0.3)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_figure_path, dpi=300)
    plt.close()
    print(f"Saved: {output_figure_path}")


def compare_image_spectral_profiles_pairs(
    image_groups_dict: dict,
    output_figure_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
    ):
    """
    Plots paired spectral profiles for before-and-after image comparisons.

    Args:
        image_groups_dict (dict): Mapping of labels to image path pairs (before, after):
            {'Image A': [
                '/image/before/a.tif',
                'image/after/a.tif'
            ],
            'Image B': [
                '/image/before/b.tif',
                '/image/after/b.tif'
            ]}
        output_figure_path (str): Path to save the resulting comparison figure.
        title (str): Title of the plot.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.

    Outputs:
        Saves a spectral comparison plot showing pre- and post-processing profiles.
    """

    os.makedirs(os.path.dirname(output_figure_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    colors = itertools.cycle(plt.cm.tab10.colors)

    for label, group in image_groups_dict.items():
        if len(group) == 2:
            image_path1, image_path2 = group
            color = next(colors)

            for i, image_path in enumerate([image_path1, image_path2]):
                with rasterio.open(image_path) as src:
                    img = src.read()
                    num_bands = img.shape[0]
                    img_reshaped = img.reshape(num_bands, -1)
                    nodata = src.nodata
                    if nodata is not None:
                        img_reshaped = np.where(img_reshaped == nodata, np.nan, img_reshaped)
                    mean_spectral = np.nanmean(img_reshaped, axis=1)
                    bands = np.arange(1, num_bands + 1)
                    linestyle = 'dashed' if i == 0 else 'solid'
                    plt.plot(bands, mean_spectral, linestyle=linestyle, color=color,
                             label=f"{label} - {'Before' if i == 0 else 'After'}")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_figure_path, dpi=300)
    plt.close()
    print(f"Saved: {output_figure_path}")


def compare_spatial_spectral_difference_band_average(
    input_images: list,
    output_image_path: str,
    title: str,
    diff_label: str,
    subtitle: str,
):
    """
    Computes and visualizes the average per-band spectral difference between two coregistered, equal size images.

    Args:
        input_images (list): List of two image file paths to compare.
        output_image_path (str): Path to save the resulting difference image (PNG).
        title (str): Title for the plot.
        diff_label (str): Label for the colorbar indicating the difference metric.
        subtitle (str): Optional subtitle to display below the plot.

    Returns:
        None
    """

    if len(input_images) != 2:
        raise ValueError("input_images must be a list of exactly two image paths.")

    path1, path2 = input_images
    name1 = os.path.splitext(os.path.basename(path1))[0]
    name2 = os.path.splitext(os.path.basename(path2))[0]

    with rasterio.open(path1) as src1, rasterio.open(path2) as src2:
        img1 = src1.read()
        img2 = src2.read()
        nodata = src1.nodata

        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions.")

        diff = np.abs(img2 - img1).astype("float32")

        if nodata is not None:
            mask = img1[0] != nodata
            for b in range(1, img1.shape[0]):
                mask &= img1[b] != nodata
                mask &= img2[b] != nodata
            diff[:, ~mask] = np.nan

        with np.errstate(invalid="ignore"):
            mean_diff = np.full(diff.shape[1:], np.nan)
            valid_mask = ~np.all(np.isnan(diff), axis=0)
            mean_diff[valid_mask] = np.nanmean(diff[:, valid_mask], axis=0)

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        im = ax.imshow(mean_diff, cmap='coolwarm', interpolation='nearest')

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(diff_label)

        ax.set_title(title, fontsize=14, pad=12)
        if subtitle:
            ax.text(0.5, -0.1, subtitle, fontsize=10, ha='center', transform=ax.transAxes)

        ax.axis("off")
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_image_path}")