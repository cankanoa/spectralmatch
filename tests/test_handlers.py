import os
import pytest

from spectralmatch import search_paths, create_paths, match_paths


@pytest.fixture
def dummy_files(tmp_path):
    paths = [
        tmp_path / "A_GlobalMatch.tif",
        tmp_path / "B_GlobalMatch.tif",
        tmp_path / "C_Other.tif",
    ]
    for p in paths:
        p.write_text("test")
    return tmp_path, [str(p) for p in paths]


# search_paths
def test_search_paths_glob(dummy_files):
    folder, all_files = dummy_files
    result = search_paths(os.path.join(str(folder), "*_GlobalMatch.tif"))
    assert len(result) == 2
    assert all("_GlobalMatch" in os.path.basename(p) for p in result)


def test_search_paths_match_to_paths(dummy_files):
    folder, all_files = dummy_files
    reference_paths = ["A", "B"]
    match_regex = r"(.*)_GlobalMatch\.tif$"
    result = search_paths(
        os.path.join(str(folder), "*_GlobalMatch.tif"),
        match_to_paths=(reference_paths, match_regex),
    )
    assert len(result) == 2
    assert all(any(r in p for r in reference_paths) for p in result)


# create_paths
def test_create_paths_from_paths(tmp_path):
    input_paths = [
        str(tmp_path / "A_GlobalMatch.tif"),
        str(tmp_path / "B_GlobalMatch.tif"),
    ]
    for p in input_paths:
        open(p, "w").close()

    output_folder = tmp_path / "out"
    template = "$_processed.tif"

    result = create_paths(os.path.join(str(output_folder), template), input_paths)
    assert len(result) == 2
    assert all(p.endswith("_processed.tif") for p in result)
    assert all(os.path.exists(os.path.dirname(p)) for p in result)


def test_create_paths_from_basenames(tmp_path):
    basenames = ["A_GlobalMatch", "B_GlobalMatch"]
    output_folder = tmp_path / "out"
    template = "$_done.tif"

    result = create_paths(os.path.join(str(output_folder), template), basenames)
    expected = [str(output_folder / f"{name}_done.tif") for name in basenames]

    assert result == expected
    assert all(os.path.exists(os.path.dirname(p)) for p in result)


def test_create_paths_disable_folder_creation(tmp_path):
    basenames = ["Image1", "Image2"]
    output_folder = tmp_path / "subdir"
    template = "$.tif"

    result = create_paths(
        os.path.join(str(output_folder), template), basenames, create_folders=False
    )
    assert all(str(output_folder) in p for p in result)
    assert not output_folder.exists()  # folder should not exist


# match_paths
def test_match_paths_success():
    input_paths = [
        "/data/A_LocalMatch.gpkg",
        "/data/B_LocalMatch.gpkg",
        "/data/C_LocalMatch.gpkg",
    ]
    reference_paths = [
        "/ref/A_GlobalMatch.tif",
        "/ref/B_GlobalMatch.tif",
        "/ref/C_GlobalMatch.tif",
    ]
    match_regex = r"(.*)_LocalMatch\.gpkg$"

    result = match_paths(input_paths, reference_paths, match_regex)
    assert result[0].endswith("A_LocalMatch.gpkg")
    assert result[1].endswith("B_LocalMatch.gpkg")
    assert result[2].endswith("C_LocalMatch.gpkg")


def test_match_paths_partial_match():
    input_paths = ["/data/B_LocalMatch.gpkg"]
    reference_paths = ["/ref/A_GlobalMatch.tif", "/ref/B_GlobalMatch.tif"]
    match_regex = r"(.*)_LocalMatch\.gpkg$"

    result = match_paths(input_paths, reference_paths, match_regex)
    assert result == [None, "/data/B_LocalMatch.gpkg"]


def test_match_paths_no_match():
    input_paths = ["/data/X_LocalMatch.gpkg"]
    reference_paths = ["/ref/Y_GlobalMatch.tif"]
    match_regex = r"(.*)_LocalMatch\.gpkg$"

    result = match_paths(input_paths, reference_paths, match_regex)
    assert result == [None]


def test_match_paths_multiple_candidates():
    input_paths = ["/data/A_LocalMatch.gpkg", "/data/A_Alt_LocalMatch.gpkg"]
    reference_paths = ["/ref/A_GlobalMatch.tif"]
    match_regex = r"(.*)_LocalMatch\.gpkg$"

    result = match_paths(input_paths, reference_paths, match_regex)
    assert result[0] in input_paths
