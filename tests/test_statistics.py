import pytest
import os

from spectralmatch import compare_image_spectral_profiles, compare_image_spectral_profiles_pairs, compare_spatial_spectral_difference_band_average
from .test_utils import create_dummy_raster


@pytest.fixture
def spectral_test_rasters(tmp_path):
    """
    Creates two dummy rasters in an input directory and an empty output directory.

    Returns:
        Tuple[dict, str]: Dictionary with labels as keys and raster paths as values, and the output directory path.
    """
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    image_dict = {}
    for label, value in [("Image A", 80), ("Image B", 160)]:
        raster_path = input_dir / f"{label.replace(' ', '_')}.tif"
        create_dummy_raster(raster_path, width=16, height=16, count=5, fill_value=value)
        image_dict[label] = str(raster_path)

    return image_dict, str(output_dir)


# compare_image_spectral_profiles_pairs
def test_compare_image_spectral_profiles(spectral_test_rasters):
    image_dict, output_dir = spectral_test_rasters
    output_path = os.path.join(output_dir, "spectral_profiles.png")

    compare_image_spectral_profiles(
        input_image_dict=image_dict,
        output_figure_path=output_path,
        title="Spectral Comparison",
        xlabel="Bands",
        ylabel="Median Value",
    )

    assert os.path.exists(output_path)


# compare_image_spectral_profiles
def test_compare_image_spectral_profiles_pairs(spectral_test_rasters):
    image_dict, output_dir = spectral_test_rasters
    pair_dict = {}

    for label, before_path in image_dict.items():
        after_filename = os.path.basename(before_path).replace(".tif", "_after.tif")
        after_path = os.path.join(os.path.dirname(before_path), after_filename)
        create_dummy_raster(after_path, width=16, height=16, count=5, fill_value=200)
        pair_dict[label] = [before_path, after_path]

    output_path = os.path.join(output_dir, "paired_profiles.png")

    compare_image_spectral_profiles_pairs(
        image_groups_dict=pair_dict,
        output_figure_path=output_path,
        title="Before vs After Comparison",
        xlabel="Band",
        ylabel="Mean Value",
    )

    assert os.path.exists(output_path)


# compare_spatial_spectral_difference_band_average
def test_compare_spatial_spectral_difference_band_average(spectral_test_rasters):
    image_dict, output_dir = spectral_test_rasters
    images = list(image_dict.values())
    assert len(images) >= 2, "Need at least two images for difference comparison"

    output_path = os.path.join(output_dir, "spatial_diff.png")

    compare_spatial_spectral_difference_band_average(
        input_images=[images[0], images[1]],
        output_image_path=output_path,
        title="Difference Map",
        diff_label="Mean Band Abs Diff",
        subtitle="Test difference between A and B",
    )

    assert os.path.exists(output_path)