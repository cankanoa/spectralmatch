import subprocess
import re
from .utils_test import create_dummy_raster


def test_cli_general_help():
    cli_function_names = [
        "align_rasters",
        "band_math",
        "compare_image_spectral_profiles",
        "compare_image_spectral_profiles_pairs",
        "compare_spatial_spectral_difference_band_average",
        "create_cloud_mask_with_omnicloudmask",
        "create_ndvi_raster",
        "global_regression",
        "local_block_adjustment",
        "mask_rasters",
        "match_paths",
        "merge_rasters",
        "merge_vectors",
        "process_raster_values_to_vector_polygons",
        "search_paths",
        "threshold_raster",
        "voronoi_center_seamline",
        "create_paths",
    ]

    result = subprocess.run(["spectralmatch", "--help"], capture_output=True, text=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0
    for name in cli_function_names:
        assert name in output, f"'{name}' not found in CLI help output"


def test_cli_command_help():
    result = subprocess.run(
        ["spectralmatch", "global_regression", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "global_regression" in (result.stdout + result.stderr)


def test_cli_version():
    result = subprocess.run(
        ["spectralmatch", "--version"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert re.search(
        r"\b\d+\.\d+\.\d+\b", result.stdout
    ), "Version number not found in output"


def test_cli_commands_run(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    input_path = input_dir / "input.tif"
    output_path_ndvi = output_dir / "ndvi.tif"
    output_path_thresh = output_dir / "thresh.tif"
    output_path_global = output_dir / "global.tif"
    create_dummy_raster(input_path, width=16, height=16, count=3)

    commands = {
        "create_ndvi_raster": [
            "spectralmatch",
            "create_ndvi_raster",
            str(input_path),
            str(output_path_ndvi),
            "3",
            "1",
        ],
        "threshold_raster": [
            "spectralmatch",
            "threshold_raster",
            "--input_images",
            str(input_path),
            "--output_images",
            str(output_path_thresh),
            "--threshold_math",
            "b1 > 0",
        ],
        "global_regression": [
            "spectralmatch",
            "global_regression",
            "--input_images",
            str(input_path),
            "--output_images",
            str(output_path_global),
        ],
    }

    for name, (cmd, out_path) in {
        "create_ndvi_raster": (commands["create_ndvi_raster"], output_path_ndvi),
        "threshold_raster": (commands["threshold_raster"], output_path_thresh),
        "global_regression": (commands["global_regression"], output_path_global),
    }.items():
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"{name} failed with error: {result.stderr}"
        assert out_path.exists(), f"{name} did not produce output."
