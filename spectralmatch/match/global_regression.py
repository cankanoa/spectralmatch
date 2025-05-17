import multiprocessing as mp
import warnings

import rasterio
import fiona
import os
import numpy as np
import sys
import json

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple, Literal

from numpy import ndarray
from scipy.optimize import least_squares
from rasterio.windows import Window
from rasterio.transform import rowcol
from rasterio.features import geometry_mask
from rasterio.coords import BoundingBox

from ..utils import _create_windows, _check_raster_requirements, _get_nodata_value, _choose_context
from ..handlers import _resolve_input_output_paths
_worker_dataset_cache = {}

# def _mask_tile(
#         src,
#         geoms,
#         window
# ):
#
#     transform = src.window_transform(window)
#     shape = (window.height, window.width)
#
#     mask_arr = geometry_mask(
#         geometries=geoms,
#         transform=transform,
#         invert=True,
#         out_shape=shape
#     )
#
#     tile = src.read(window=window)
#     return tile * mask_arr.astype(tile.dtype)

def global_regression(
    input_images: str | List[str],
    output_images: Tuple[str, str] | List[str],
    *,
    custom_mean_factor: float = 1.0,
    custom_std_factor: float = 1.0,
    vector_mask_path: Optional[str] = None,
    window_size: int | Tuple[int, int] | None = None,
    debug_logs: bool = False,
    custom_nodata_value: float | None = None,
    parallel_workers: Literal["cpu"] | int | None = None,
    calculation_dtype_precision: str = "float32",
    specify_model_images: Tuple[Literal["exclude", "include"], List[str]] | None = None,
    save_adjustments: str | None = None,
    load_adjustments: str | None = None,
    ) -> list:
    """
    Performs global radiometric normalization across overlapping images using least squares regression.

    Args:
        input_images (str | List[str]): A folder path containing `.tif` files to search for or a list of input image paths.
        output_images (Tuple[str, str] | List[str]): Either a tuple of (output_folder, suffix) to generate output paths from, or a list of output image paths. If a list is provided, its length must match the number of input images.
        custom_mean_factor (float, optional): Weight for mean constraints in regression. Defaults to 1.0.
        custom_std_factor (float, optional): Weight for standard deviation constraints in regression. Defaults to 1.0.
        vector_mask_path (Optional[str], optional): Optional mask to limit stats to specific areas. Defaults to None.
        window_size (int | Tuple[int, int] | None): Tile size for processing: int for square tiles, (width, height) for custom size, or None for full image. Defaults to None.
        debug_logs (bool, optional): If True, prints debug information and constraint matrices. Defaults to False.
        custom_nodata_value (float | None, optional): Overrides detected NoData value. Defaults to None.
        parallel_workers (Literal["cpu"] | int | None): If set, enables multiprocessing. "cpu" = all cores, int = specific count, None = no parallel processing. Defaults to None.
        calculation_dtype_precision (str, optional): Data type used for internal calculations. Defaults to "float32".
        specify_model_images (Tuple[Literal["exclude", "include"], List[str]] | None ): First item in tuples sets weather to 'include' or 'exclude' the listed images from model building statistics. Second item is the list of image names (without their extension) to apply criteria to. For example, if this param is only set to 'include' one image, all other images will be matched to that one image. Defaults to no exclusion.
        save_adjustments (str | None, optional): The output path of a .json file to save adjustments parameters. Defaults to not saving.
        load_adjustments (str | None, optional): If set, loads saved whole and overlapping statistics only for images that exist in the .json file. Other images will still have their statistics calculated. Defaults to None.

    Returns:
        List[str]: Paths to the globally adjusted output raster images.
    """

    print("Start global regression")

    input_image_paths, output_image_paths = _resolve_input_output_paths(input_images, output_images)
    input_image_names = list(input_image_paths.keys())
    num_input_images = len(input_image_paths)


    _check_raster_requirements(list(input_image_paths.values()), debug_logs)

    if isinstance(window_size, int): window_size = (window_size, window_size)
    nodata_val = _get_nodata_value(list(input_image_paths.values()), custom_nodata_value)

    # Find loaded and input files if load adjustments
    loaded_model = {}
    if load_adjustments:
        with open(load_adjustments, "r") as f:
            loaded_model = json.load(f)
        _validate_adjustment_model_structure(loaded_model)
        loaded_names = set(loaded_model.keys())
        input_names = set(input_image_names)
    else:
        loaded_names = set([])
        input_names = set(input_image_names)

    matched = input_names & loaded_names
    only_loaded = loaded_names - input_names
    only_input = input_names - loaded_names
    if debug_logs:
        print(f"Total images: input images: {len(input_names)}, loaded images {len(loaded_names)}: ")
        print(f"    Matched adjustments (to override) ({len(matched)}):", sorted(matched))
        print(f"    Only in loaded adjustments (to add) ({len(only_loaded)}):", sorted(only_loaded))
        print(f"    Only in input (to calculate) ({len(only_input)}):", sorted(only_input))

    # Find images to include in model
    included_names = list(matched | only_loaded | only_input)
    if specify_model_images:
        mode, names = specify_model_images
        name_set = set(names)
        if mode == "include":
            included_names = [n for n in input_image_names if n in name_set]
        elif mode == "exclude":
            included_names = [n for n in input_image_names if n not in name_set]
        excluded_names = [n for n in input_image_names if n not in included_names]
    if debug_logs:
        print("Images to influence the model:")
        print(f"    Included in model ({len(included_names)}): {sorted(included_names)}")
        if specify_model_images: print(f"    Excluded from model ({len(excluded_names)}): {sorted(excluded_names)}")
        else: print(f"    Excluded from model (0): []")

    # Calculate stats
    if debug_logs: print("Calculating statistics")
    with rasterio.open(list(input_image_paths.values())[0]) as src: num_bands = src.count

    # Get images bounds
    all_bounds = {}
    for name, path in input_image_paths.items():
        with rasterio.open(path) as ds:
            all_bounds[name] = ds.bounds

    # Calculate overlap stats
    overlapping_pairs = _find_overlaps(all_bounds)

    all_overlap_stats = {}
    for name_i, name_j in overlapping_pairs:

        if name_i in loaded_model and name_j in loaded_model[name_i].get("overlap_stats", {}):
            continue

        stats = _calculate_overlap_stats(
            num_bands,
            input_image_paths[name_i],
            input_image_paths[name_j],
            name_i,
            name_j,
            all_bounds[name_i],
            all_bounds[name_j],
            nodata_val,
            nodata_val,
            vector_mask_path=vector_mask_path,
            window_size=window_size,
            debug_logs=debug_logs,
        )

        for k_i, v in stats.items():
            all_overlap_stats[k_i] = {
                **all_overlap_stats.get(k_i, {}),
                **{
                    k_j: {**all_overlap_stats.get(k_i, {}).get(k_j, {}), **s}
                    for k_j, s in v.items()
                },
            }


    # Add loaded image stats to model
    if load_adjustments:
        for name_i, model_entry in loaded_model.items():
            if name_i not in input_image_paths:
                continue

            for name_j, bands in model_entry.get("overlap_stats", {}).items():
                if name_j not in input_image_paths:
                    continue

                all_overlap_stats.setdefault(name_i, {})[name_j] = {
                    int(k.split("_")[1]): {
                        "mean": bands[k]["mean"],
                        "std": bands[k]["std"],
                        "size": bands[k]["size"]
                    } for k in bands
                }

    # Calculate whole stats
    all_whole_stats = {}
    for name, path in input_image_paths.items():
        if name in loaded_model:
            all_whole_stats[name] = {
                int(k.split("_")[1]): {
                    "mean": loaded_model[name]["whole_stats"][k]["mean"],
                    "std": loaded_model[name]["whole_stats"][k]["std"],
                    "size": loaded_model[name]["whole_stats"][k]["size"]
                }
                for k in loaded_model[name]["whole_stats"]
            }
        else:
            all_whole_stats.update(
                _calculate_whole_stats(
                    input_image_path=path,
                    nodata=nodata_val,
                    num_bands=num_bands,
                    image_name=name,
                    vector_mask_path=vector_mask_path,
                    window_size=window_size,
                )
            )

    all_image_names = list(dict.fromkeys(input_image_names + list(loaded_model.keys())))
    num_total = len(all_image_names)

    # Print model sources
    if debug_logs:
        print(f"\nCreating model for {len(all_image_names)} total images from {len(included_names)} included:")
        print(f"{'ID':<4}\t{'Source':<6}\t{'Inclusion':<8}\tName")
        for i, name in enumerate(all_image_names):
            source = "load" if name in (matched | only_loaded) else "calc"
            included = "incl" if name in included_names else "excl"
            print(f"{i:<4}\t{source:<6}\t{included:<8}\t{name}")

    # Build model
    all_params = np.zeros((num_bands, 2 * num_total, 1), dtype=float)
    image_names_with_id = [(i, name) for i, name in enumerate(all_image_names)]
    for b in range(num_bands):
        if debug_logs: print(f"\nProcessing band {b}:")

        A, y, tot_overlap = [], [], 0
        for i, name_i in image_names_with_id:
            for j, name_j in image_names_with_id[i + 1:]:
                stat = all_overlap_stats.get(name_i, {}).get(name_j)
                if stat is None:
                    continue

                # This condition ensures that only overlaps involving at least one included image contribute constraints, allowing external images to be calibrated against the model without influencing it.
                if name_i not in included_names and name_j not in included_names:
                    continue

                s = stat[b]["size"]
                m1, v1 = stat[b]["mean"], stat[b]["std"]
                m2, v2 = (
                    all_overlap_stats[name_j][name_i][b]["mean"],
                    all_overlap_stats[name_j][name_i][b]["std"],
                )

                row_m = [0] * (2 * num_total)
                row_s = [0] * (2 * num_total)
                row_m[2 * i: 2 * i + 2] = [m1, 1]
                row_m[2 * j: 2 * j + 2] = [-m2, -1]
                row_s[2 * i], row_s[2 * j] = v1, -v2

                A.extend([
                    [v * s * custom_mean_factor for v in row_m],
                    [v * s * custom_std_factor for v in row_s],
                ])
                y.extend([0, 0])
                tot_overlap += s

        pjj = 1.0 if tot_overlap == 0 else tot_overlap / (2.0 * num_total)

        for name in included_names:
            mj = all_whole_stats[name][b]["mean"]
            vj = all_whole_stats[name][b]["std"]
            j_idx = all_image_names.index(name)
            row_m = [0] * (2 * num_total)
            row_s = [0] * (2 * num_total)
            row_m[2 * j_idx: 2 * j_idx + 2] = [mj * pjj, 1 * pjj]
            row_s[2 * j_idx] = vj * pjj
            A.extend([row_m, row_s])
            y.extend([mj * pjj, vj * pjj])

        for name in input_image_names:
            if name in included_names:
                continue
            row = [0] * (2 * num_total)
            A.append(row.copy())
            y.append(0)
            A.append(row.copy())
            y.append(0)

        A_arr = np.asarray(A)
        y_arr = np.asarray(y)
        res = least_squares(lambda p: A_arr @ p - y_arr, [1, 0] * num_total)

        if debug_logs:
            _print_constraint_system(
                constraint_matrix=A_arr,
                adjustment_params=res.x,
                observed_values_vector=y_arr,
                overlap_pairs=overlapping_pairs,
                image_names_with_id=image_names_with_id,

            )

        all_params[b] = res.x.reshape((2 * num_total, 1))

    # Save adjustments
    if save_adjustments:
        _save_adjustments(
            save_path=save_adjustments,
            input_image_names=list(input_image_paths.keys()),
            all_params=all_params,
            all_whole_stats=all_whole_stats,
            all_overlap_stats=all_overlap_stats,
            num_bands=num_bands,
            calculation_dtype_precision=calculation_dtype_precision
        )

    if parallel_workers == "cpu":
        parallel = True
        max_workers = mp.cpu_count()
    elif isinstance(parallel_workers, int) and parallel_workers > 0:
        parallel = True
        max_workers = parallel_workers
    else:
        parallel = False
        max_workers = None

    out_paths: List[str] = []
    for idx, (name, img_path) in enumerate(input_image_paths.items()):
        out_path = output_image_paths[name]
        out_paths.append(out_path)

        if debug_logs: print(f"Apply adjustments and saving results for {name}")
        with rasterio.open(img_path) as src:
            meta = src.meta.copy()
            meta.update({"count": num_bands, "nodata": nodata_val})
            with rasterio.open(out_path, "w", **meta) as dst:

                if window_size:
                    tw, th = window_size
                    windows = list(_create_windows(src.width, src.height, tw, th))
                else:
                    windows = [Window(0, 0, src.width, src.height)]

                if parallel:
                    ctx = _choose_context(prefer_fork=True)
                    pool = ProcessPoolExecutor(
                        max_workers=max_workers,
                        mp_context=ctx,
                        initializer=_init_worker,
                        initargs=(img_path,),
                    )

                for b in range(num_bands):
                    a = float(all_params[b, 2 * idx, 0])
                    b0 = float(all_params[b, 2 * idx + 1, 0])

                    if parallel:
                        futs = [
                            pool.submit(_process_tile_global,
                                        w,
                                        b,
                                        a,
                                        b0,
                                        nodata_val,
                                        calculation_dtype_precision,
                                        debug_logs,
                                        )
                            for w in windows
                        ]
                        for fut in as_completed(futs):
                            win, buf = fut.result()
                            dst.write(buf.astype(meta["dtype"]), b + 1, window=win)
                    else:
                        for win in windows:
                            _, buf = _process_tile_global(
                                win,
                                b,
                                a,
                                b0,
                                nodata_val,
                                debug_logs,
                            )
                            dst.write(buf.astype(meta["dtype"]), b + 1, window=win)
                if parallel:
                    pool.shutdown()

    print("Finished global regression")
    return out_paths


def _save_adjustments(
    save_path: str,
    input_image_names: List[str],
    all_params: np.ndarray,
    all_whole_stats: dict,
    all_overlap_stats: dict,
    num_bands: int,
    calculation_dtype_precision: str
    ) -> None:
    """
    Saves adjustment parameters, whole-image stats, and overlap stats in a nested JSON format.

    Args:
        save_path (str): Output JSON path.
        input_image_names (List[str]): List of input image names.
        all_params (np.ndarray): Adjustment parameters, shape (bands, 2 * num_images, 1).
        all_whole_stats (dict): Per-image stats (keyed by image name).
        all_overlap_stats (dict): Per-pair overlap stats (keyed by image name).
        num_bands (int): Number of bands.
        calculation_dtype_precision (str): Precision for saving values (e.g., "float32").
    """

    if not os.path.exists(os.path.dirname(save_path)): os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cast = lambda x: float(np.dtype(calculation_dtype_precision).type(x))

    full_model = {}
    for i, name in enumerate(input_image_names):
        full_model[name] = {
            "adjustments": {
                f"band_{b}": {
                    "scale": cast(all_params[b, 2 * i, 0]),
                    "offset": cast(all_params[b, 2 * i + 1, 0])
                } for b in range(num_bands)
            },
            "whole_stats": {
                f"band_{b}": {
                    "mean": cast(all_whole_stats[name][b]["mean"]),
                    "std": cast(all_whole_stats[name][b]["std"]),
                    "size": int(all_whole_stats[name][b]["size"])
                } for b in range(num_bands)
            },
            "overlap_stats": {}
        }

    for name_i, j_stats in all_overlap_stats.items():
        for name_j, band_stats in j_stats.items():
            if name_j not in full_model[name_i]["overlap_stats"]:
                full_model[name_i]["overlap_stats"][name_j] = {}
            for b, stats in band_stats.items():
                full_model[name_i]["overlap_stats"][name_j][f"band_{b}"] = {
                    "mean": cast(stats["mean"]),
                    "std": cast(stats["std"]),
                    "size": int(stats["size"])
                }

    with open(save_path, "w") as f:
        json.dump(full_model, f, indent=2)

def _validate_adjustment_model_structure(model: dict) -> None:
    """
    Validates the structure of a loaded adjustment model dictionary.

    Ensures that:
    - Each top-level key is an image name mapping to a dictionary.
    - Each image has 'adjustments' and 'whole_stats' with per-band keys like 'band_0'.
    - Each band entry in 'adjustments' contains 'scale' and 'offset'.
    - Each band entry in 'whole_stats' contains 'mean', 'std', and 'size'.
    - If present, 'overlap_stats' maps to other image names with valid per-band statistics.

    The expected model structure is a dictionary with this format:

    {
        "image_name_1": {
            "adjustments": {
                "band_0": {"scale": float, "offset": float},
                "band_1": {"scale": float, "offset": float},
                ...
            },
            "whole_stats": {
                "band_0": {"mean": float, "std": float, "size": int},
                "band_1": {"mean": float, "std": float, "size": int},
                ...
            },
            "overlap_stats": {
                "image_name_2": {
                    "band_0": {"mean": float, "std": float, "size": int},
                    "band_1": {"mean": float, "std": float, "size": int},
                    ...
                },
                ...
            }
        },
        ...
    }

    - Keys are image basenames (without extension).
    - Band keys are of the form "band_0", "band_1", etc.
    - All numerical values are stored as floats (except 'size', which is an int).

    Args:
        model (dict): Parsed JSON adjustment model.

    Raises:
        ValueError: If any structural issues or missing keys are detected.
    """
    for image_name, image_data in model.items():
        if not isinstance(image_data, dict):
            raise ValueError(f"'{image_name}' must map to a dictionary.")

        adjustments = image_data.get("adjustments")
        if not isinstance(adjustments, dict):
            raise ValueError(f"'{image_name}' is missing 'adjustments' dictionary.")

        for band_key, band_vals in adjustments.items():
            if not band_key.startswith("band_"):
                raise ValueError(f"Invalid band key '{band_key}' in adjustments for '{image_name}'.")
            if not {"scale", "offset"} <= band_vals.keys():
                raise ValueError(f"Missing 'scale' or 'offset' in adjustments[{band_key}] for '{image_name}'.")

        whole_stats = image_data.get("whole_stats")
        if not isinstance(whole_stats, dict):
            raise ValueError(f"'{image_name}' is missing 'whole_stats' dictionary.")

        for band_key, stat_vals in whole_stats.items():
            if not band_key.startswith("band_"):
                raise ValueError(f"Invalid band key '{band_key}' in whole_stats for '{image_name}'.")
            if not {"mean", "std", "size"} <= stat_vals.keys():
                raise ValueError(f"Missing 'mean', 'std', or 'size' in whole_stats[{band_key}] for '{image_name}'.")

        overlap_stats = image_data.get("overlap_stats", {})
        if not isinstance(overlap_stats, dict):
            raise ValueError(f"'overlap_stats' for '{image_name}' must be a dictionary if present.")

        for other_image, bands in overlap_stats.items():
            if not isinstance(bands, dict):
                raise ValueError(f"'overlap_stats[{other_image}]' for '{image_name}' must be a dictionary.")
            for band_key, stat_vals in bands.items():
                if not band_key.startswith("band_"):
                    raise ValueError(f"Invalid band key '{band_key}' in overlap_stats[{other_image}] for '{image_name}'.")
                if not {"mean", "std", "size"} <= stat_vals.keys():
                    raise ValueError(f"Missing 'mean', 'std', or 'size' in overlap_stats[{other_image}][{band_key}] for '{image_name}'.")
    print("Loaded adjustments structure passed validation")


def _process_tile_global(
    window: Window,
    band_idx: int,
    a: float,
    b: float,
    nodata: int | float,
    calculation_dtype_precision: str,
    debug_logs: bool,
    ):
    """
    Applies a global linear transformation (scale and offset) to a raster tile.

    Args:
        window (Window): Rasterio window specifying the region to process.
        band_idx (int): Band index to read and adjust.
        a (float): Multiplicative factor for normalization.
        b (float): Additive offset for normalization.
        nodata (int | float): NoData value to ignore during processing.
        calculation_dtype_precision (str): Data type to cast the block for computation.
        debug_logs (bool): If True, prints processing information.

    Returns:
        Tuple[Window, np.ndarray]: Window and the adjusted data block.
    """

    # if debug_logs: print(f"Processing band: {band_idx}, window: {window}")
    ds = _worker_dataset_cache["ds"]
    block = ds.read(band_idx + 1, window=window).astype(calculation_dtype_precision)

    mask = block != nodata
    block[mask] = a * block[mask] + b
    return window, block


def _print_constraint_system(
    constraint_matrix: np.ndarray,
    adjustment_params: np.ndarray,
    observed_values_vector: np.ndarray,
    overlap_pairs: tuple,
    image_names_with_id: list[tuple[int, str]],
) -> None:
    """
    Prints the constraint matrix system with labeled rows and columns for debugging regression inputs.

    Args:
        constraint_matrix (ndarray): Coefficient matrix used in the regression system.
        adjustment_params (ndarray): Solved adjustment parameters (regression output).
        observed_values_vector (ndarray): Target values in the regression system.
        overlap_pairs (tuple): Pairs of overlapping image indices used in constraints.
        image_names_with_id (list of tuple): List of (ID, name) pairs corresponding to each image's position in the system.

    Returns:
        None
    """
    np.set_printoptions(
        suppress=True,
        precision=3,
        linewidth=300,
        formatter={"float_kind": lambda x: f"{x: .3f}"},
    )

    print("constraint_matrix with labels:")

    name_to_id = {n: i for i, n in image_names_with_id}

    # Build row labels
    row_labels = []
    for i, j in overlap_pairs:
        row_labels.append(f"Overlap({name_to_id[i]}-{name_to_id[j]}) Mean Diff")
        row_labels.append(f"Overlap({name_to_id[i]}-{name_to_id[j]}) Std Diff")

    for i, name in image_names_with_id:
        row_labels.append(f"[{i}] Mean Cnstr")
        row_labels.append(f"[{i}] Std Cnstr")

    # Build column labels
    col_labels = []
    for i, name in image_names_with_id:
        col_labels.append(f"a{i}")
        col_labels.append(f"b{i}")

    # Print column headers
    header = f"{'':<30}"
    for lbl in col_labels:
        header += f"{lbl:>18}"
    print(header)

    # Print matrix rows
    for row_label, row in zip(row_labels, constraint_matrix):
        line = f"{row_label:<30}"
        for val in row:
            line += f"{val:18.3f}"
        print(line)

    print("\nadjustment_params:")
    np.savetxt(sys.stdout, adjustment_params, fmt="%18.3f")

    print("\nobserved_values_vector:")
    np.savetxt(sys.stdout, observed_values_vector, fmt="%18.3f")


def _find_overlaps(
    image_bounds_dict: dict[str, rasterio.coords.BoundingBox]
    ) -> tuple[tuple[str, str], ...]:
    """
    Finds all pairs of image names with overlapping spatial bounds.

    Args:
        image_bounds_dict (dict): Dictionary mapping image names to their rasterio bounds.

    Returns:
        Tuple[Tuple[str, str], ...]: Pairs of image names with overlapping extents.
    """
    overlaps = []

    keys = sorted(image_bounds_dict)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            b1, b2 = image_bounds_dict[k1], image_bounds_dict[k2]

            if (
                b1.left < b2.right and b1.right > b2.left and
                b1.bottom < b2.top and b1.top > b2.bottom
            ):
                overlaps.append((k1, k2))

    return tuple(overlaps)


def _calculate_overlap_stats(
    num_bands: int,
    input_image_path_i: str,
    input_image_path_j: str,
    id_i: int,
    id_j: int,
    bound_i: BoundingBox,
    bound_j: BoundingBox,
    nodata_i: int | float,
    nodata_j: int | float,
    vector_mask_path: str=None,
    window_size: tuple = None,
    debug_logs: bool =False,
    ):
    """
    Calculates mean, standard deviation, and valid pixel count for overlapping regions between two images.

    Args:
        num_bands (int): Number of bands in the images.
        input_image_path_i (str): Path to the first image.
        input_image_path_j (str): Path to the second image.
        id_i (int): Index of the first image.
        id_j (int): Index of the second image.
        bound_i (BoundingBox): Spatial bounds of the first image.
        bound_j (BoundingBox): Spatial bounds of the second image.
        nodata_i (int | float): NoData value for the first image.
        nodata_j (int | float): NoData value for the second image.
        vector_mask_path (str, optional): Optional path to a vector mask for clipping. Defaults to None.
        window_size (tuple, optional): Optional tile size for chunked processing. Defaults to None.
        debug_logs (bool, optional): If True, prints debug information. Defaults to False.

    Returns:
        dict: Nested dictionary of overlap statistics indexed by image ID and band.
    """

    stats = {id_i: {id_j: {}}, id_j: {id_i: {}}}

    with rasterio.open(input_image_path_i) as src_i, rasterio.open(input_image_path_j) as src_j:
        if vector_mask_path:
            with fiona.open(vector_mask_path, "r") as vector:
                geoms = [feat["geometry"] for feat in vector]
        else:
            geoms = None

        transform_i = src_i.transform
        transform_j = src_j.transform

        x_min = max(bound_i.left, bound_j.left)
        x_max = min(bound_i.right, bound_j.right)
        y_min = max(bound_i.bottom, bound_j.bottom)
        y_max = min(bound_i.top, bound_j.top)

        if debug_logs: print(f"Overlap bounds: x: {x_min:.2f} to {x_max:.2f}, y: {y_min:.2f} to {y_max:.2f}")

        if x_min >= x_max or y_min >= y_max:
            return stats

        row_min_i, col_min_i = rowcol(transform_i, x_min, y_max)
        row_max_i, col_max_i = rowcol(transform_i, x_max, y_min)
        row_min_j, col_min_j = rowcol(transform_j, x_min, y_max)
        row_max_j, col_max_j = rowcol(transform_j, x_max, y_min)

        height = min(row_max_i - row_min_i, row_max_j - row_min_j)
        width = min(col_max_i - col_min_i, col_max_j - col_min_j)

        if debug_logs:
            print(f"For overlap {os.path.basename(input_image_path_i)} with {os.path.basename(input_image_path_j)}:")
            print(f"    Pixel window {os.path.basename(input_image_path_i)}: cols: {col_min_i} to {col_max_i} ({col_max_i - col_min_i}), rows: {row_min_i} to {row_max_i} ({row_max_i - row_min_i})")
            print(f"    Pixel window {os.path.basename(input_image_path_j)}: cols: {col_min_j} to {col_max_j} ({col_max_j - col_min_j}), rows: {row_min_j} to {row_max_j} ({row_max_j - row_min_j})")

        for band in range(num_bands):
            if window_size:
                windows = _create_windows(width, height, window_size[0], window_size[1])
            else:
                windows = [Window(0, 0, width, height)]
            windows = _adjust_size_of_tiles_to_fit_bounds(windows, width, height)

            combined_i = []
            combined_j = []

            for win in windows:
                offset_to_window_i = Window(col_min_i + win.col_off, row_min_i + win.row_off, win.width, win.height)
                offset_to_window_j = Window(col_min_j + win.col_off, row_min_j + win.row_off, win.width, win.height)

                block_i = src_i.read(band + 1, window=offset_to_window_i)
                block_j = src_j.read(band + 1, window=offset_to_window_j)

                if geoms:
                    transform_i_win = src_i.window_transform(offset_to_window_i)
                    transform_j_win = src_j.window_transform(offset_to_window_j)

                    mask_i = geometry_mask(geoms, transform=transform_i_win, invert=True,
                                           out_shape=(int(offset_to_window_i.height), int(offset_to_window_i.width)))
                    mask_j = geometry_mask(geoms, transform=transform_j_win, invert=True,
                                           out_shape=(int(offset_to_window_j.height), int(offset_to_window_j.width)))

                    block_i[~mask_i] = nodata_i
                    block_j[~mask_j] = nodata_j

                valid = (block_i != nodata_i) & (block_j != nodata_j)
                if np.any(valid):
                    combined_i.append(block_i[valid])
                    combined_j.append(block_j[valid])

            v_i = np.concatenate(combined_i) if combined_i else np.array([])
            v_j = np.concatenate(combined_j) if combined_j else np.array([])

            stats[id_i][id_j][band] = {
                "mean": float(np.mean(v_i)) if v_i.size else 0,
                "std": float(np.std(v_i)) if v_i.size else 0,
                "size": int(v_i.size),
            }
            stats[id_j][id_i][band] = {
                "mean": float(np.mean(v_j)) if v_j.size else 0,
                "std": float(np.std(v_j)) if v_j.size else 0,
                "size": int(v_j.size),
            }
    return stats


def _adjust_size_of_tiles_to_fit_bounds(
    windows: Window,
    max_width: int,
    max_height: int,
    ):
    """
    Adjusts a list of raster windows to ensure they fit within specified maximum bounds.

    Args:
        windows (Window): Iterable of rasterio Windows to be adjusted.
        max_width (int): Maximum allowed width (in pixels).
        max_height (int): Maximum allowed height (in pixels).

    Returns:
        list[Window]: List of adjusted windows clipped to the specified bounds.
    """

    adjusted_windows = []
    for win in windows:
        new_width = min(win.width, max_width - win.col_off)
        new_height = min(win.height, max_height - win.row_off)

        if new_width > 0 and new_height > 0:
            adjusted_windows.append(Window(win.col_off, win.row_off, new_width, new_height))

    return adjusted_windows


def _calculate_whole_stats(
    input_image_path: str,
    nodata: int | float,
    num_bands: int,
    image_name: str,
    vector_mask_path: str=None,
    window_size: tuple = None,
    ):
    """
    Computes mean, standard deviation, and valid pixel count for each band in a single image.

    Args:
        input_image_path (str): Path to the input raster image.
        nodata (int | float): NoData value to ignore during calculations.
        num_bands (int): Number of bands to process.
        image_name (str): Unique name for the image, used as a key in the output.
        vector_mask_path (str, optional): Optional vector mask path to restrict statistics to masked regions. Defaults to None.
        window_size (tuple, optional): Optional tile size for block-wise processing. Defaults to None.

    Returns:
        dict: Dictionary with image ID as key and per-band statistics as sub-dictionary.
    """

    stats = {image_name: {}}

    with rasterio.open(input_image_path) as data:
        # Load geometries once if needed
        if vector_mask_path:
            with fiona.open(vector_mask_path, "r") as vector:
                geoms = [feat["geometry"] for feat in vector]
        else:
            geoms = None

        for band_idx in range(num_bands):
            mean = 0.0
            M2 = 0.0
            count = 0

            if window_size:
                windows = _create_windows(data.width, data.height, window_size[0], window_size[1])
            else:
                windows = [Window(0, 0, data.width, data.height)]

            for win in windows:
                block = data.read(band_idx + 1, window=win)

                if geoms:
                    transform = data.window_transform(win)
                    mask = geometry_mask(
                        geoms,
                        transform=transform,
                        invert=True,
                        out_shape=(int(win.height), int(win.width))
                    )
                    block[~mask] = nodata

                valid = block != nodata
                values = block[valid]

                for x in values.ravel():
                    count += 1
                    delta = x - mean
                    mean += delta / count
                    M2 += delta * (x - mean)

            if count > 1:
                variance = M2 / (count - 1)
                std = np.sqrt(variance)
            else:
                mean = 0.0
                std = 0.0

            stats[image_name][band_idx] = {
                "mean": float(mean),
                "std": float(std),
                "size": count,
            }

    return stats


def _init_worker(img_path: str):
    """
    Initializes a global dataset cache for a worker process by opening a raster file.

    Args:
        img_path (str): Path to the image file to be opened and cached.

    Returns:
        None
    """

    import rasterio
    global _worker_dataset_cache
    _worker_dataset_cache["ds"] = rasterio.open(img_path, "r")