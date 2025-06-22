import itertools
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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
                        img_reshaped = np.where(
                            img_reshaped == nodata, np.nan, img_reshaped
                        )
                    mean_spectral = np.nanmean(img_reshaped, axis=1)
                    bands = np.arange(1, num_bands + 1)
                    linestyle = "dashed" if i == 0 else "solid"
                    plt.plot(
                        bands,
                        mean_spectral,
                        linestyle=linestyle,
                        color=color,
                        label=f"{label} - {'Before' if i == 0 else 'After'}",
                    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_figure_path, dpi=300)
    plt.close()
    print(f"Saved: {os.path.splitext(os.path.basename(output_figure_path))[0]}")


def compare_spatial_spectral_difference_band_average(
    input_images: list,
    output_figure_path: str,
    title: str,
    diff_label: str,
    subtitle: str,
    scale: tuple = None,
):
    """
    Computes and visualizes the mean per-pixel spectral difference between two coregistered, equal-size images.

    Args:
        input_images (list): List of two image file paths [before, after].
        output_figure_path (str): Path to save the resulting difference image (PNG).
        title (str): Title for the plot.
        diff_label (str): Label for the colorbar.
        subtitle (str): Subtitle text shown below the image.
        scale (tuple, optional): Tuple (vmin, vmax) to fix the color scale. Centered at 0.

    Raises:
        ValueError: If the input list doesn't contain exactly two image paths, or shapes mismatch.
    """
    if len(input_images) != 2:
        raise ValueError("input_images must be a list of exactly two image paths.")

    path1, path2 = input_images

    with rasterio.open(path1) as src1, rasterio.open(path2) as src2:
        img1 = src1.read().astype("float32")
        img2 = src2.read().astype("float32")
        nodata = src1.nodata

        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions.")

        diff = img2 - img1

        if nodata is not None:
            mask = np.full(diff.shape[1:], True)
            for b in range(diff.shape[0]):
                mask &= (img1[b] != nodata) & (img2[b] != nodata)
            diff[:, ~mask] = np.nan

        with np.errstate(invalid="ignore"):
            mean_diff = np.full(diff.shape[1:], np.nan)
            valid_mask = ~np.all(np.isnan(diff), axis=0)
            mean_diff[valid_mask] = np.nanmean(diff[:, valid_mask], axis=0)

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

        vmin, vmax = scale if scale else (np.nanmin(mean_diff), np.nanmax(mean_diff))
        max_abs = max(abs(vmin), abs(vmax))
        im = ax.imshow(mean_diff, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(diff_label)

        ax.set_title(title, fontsize=14, pad=12)
        if subtitle:
            ax.text(
                0.5, -0.1, subtitle, fontsize=10, ha="center", transform=ax.transAxes
            )

        ax.axis("off")
        plt.savefig(output_figure_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {os.path.splitext(os.path.basename(output_figure_path))[0]}")


def compare_before_after_all_images(
    input_images_1: list,
    input_images_2: list,
    output_figure_path: str,
    title: str,
    ylabel_1: str,
    ylabel_2: str,
    image_names: list = None,
):
    """
    Creates a 2-row grid comparing before and after images with per-band contrast stretch and nodata transparency.

    Args:
        input_images_1 (list): Paths to "before" images (top row).
        input_images_2 (list): Paths to "after" images (bottom row).
        output_figure_path (str): File path to save the output figure.
        image_names (list, optional): Column titles. Must match image count if provided.
        title (str): Figure title.
        ylabel_1 (str): Label for the top row.
        ylabel_2 (str): Label for the bottom row.

    Raises:
        AssertionError: If input lengths mismatch or if image_names is invalid.

    Saves:
        A PNG figure with transparent nodata and matched image pairs.
    """

    assert len(input_images_1) == len(
        input_images_2
    ), "Image lists must be the same length."
    if image_names:
        assert len(image_names) == len(
            input_images_1
        ), "image_names must match image count."

    os.makedirs(os.path.dirname(output_figure_path), exist_ok=True)
    num_images = len(input_images_1)
    fig = plt.figure(figsize=(5 * num_images, 10))
    gs = gridspec.GridSpec(2, num_images + 1, width_ratios=[0.05] + [1] * num_images)

    for col_idx, (path1, path2) in enumerate(zip(input_images_1, input_images_2)):
        for row_idx, path in enumerate([path1, path2]):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])  # shift right by 1 column

            with rasterio.open(path) as src:
                nodata = src.nodata
                img = (
                    src.read([1, 2, 3])
                    if src.count >= 3
                    else np.repeat(src.read(1)[np.newaxis, ...], 3, axis=0)
                )
                img = img.astype("float32")

                mask = np.full(img.shape[1:], False)
                if nodata is not None:
                    for b in range(img.shape[0]):
                        mask |= img[b] == nodata

                for b in range(img.shape[0]):
                    valid = img[b][~mask]
                    if valid.size > 0:
                        vmin, vmax = np.percentile(valid, (2, 98))
                        img[b] = np.clip((img[b] - vmin) / (vmax - vmin), 0, 1)
                    else:
                        img[b] = 0

                img = img.transpose(1, 2, 0)
                alpha = (~mask).astype("float32")
                rgba = np.dstack((img, alpha))

                ax.imshow(rgba)
                if row_idx == 0 and image_names:
                    ax.set_title(image_names[col_idx])
                ax.axis("off")

    # Add vertical row labels
    for i, label in enumerate([ylabel_1, ylabel_2]):
        ax = fig.add_subplot(gs[i, 0])
        ax.set_ylabel(label, fontsize=12, rotation=90, labelpad=10, va="center")
        ax.tick_params(left=False, labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_figure_path, dpi=300)
    plt.close()
    print(f"Saved: {os.path.splitext(os.path.basename(output_figure_path))[0]}")
