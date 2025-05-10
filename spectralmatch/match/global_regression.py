import multiprocessing as mp
import rasterio
import fiona
import os
import numpy as np
import sys

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

from numpy import ndarray
from scipy.optimize import least_squares
from rasterio.windows import Window
from rasterio.transform import rowcol
from rasterio.features import geometry_mask
from rasterio.coords import BoundingBox

from ..utils import _create_windows, _check_raster_requirements, _get_nodata_value, _choose_context
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
    input_image_paths: List[str],
    output_image_folder: str,
    *,
    custom_mean_factor: float = 1.0,
    custom_std_factor: float = 1.0,
    output_global_basename: str = "_global",
    vector_mask_path: Optional[str] = None,
    window_size: Optional[Tuple[int, int]] = None,
    debug_mode: bool = False,
    custom_nodata_value: float | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
    calculation_dtype_precision: str = "float32",
    ) -> list:
    """
    Performs global radiometric normalization across overlapping images using least squares regression.

    Args:
        input_image_paths (List[str]): List of input raster image paths.
        output_image_folder (str): Folder to save normalized output images.
        custom_mean_factor (float, optional): Weight for mean constraints in regression. Defaults to 1.0.
        custom_std_factor (float, optional): Weight for standard deviation constraints in regression. Defaults to 1.0.
        output_global_basename (str, optional): Suffix for output filenames. Defaults to "_global".
        vector_mask_path (Optional[str], optional): Optional mask to limit stats to specific areas. Defaults to None.
        window_size (Optional[Tuple[int, int]], optional): Tile size for block-wise processing. Defaults to None.
        debug_mode (bool, optional): If True, prints debug information and constraint matrices. Defaults to False.
        custom_nodata_value (float | None, optional): Overrides detected NoData value. Defaults to None.
        parallel (bool, optional): Enables parallel tile processing. Defaults to False.
        max_workers (int | None, optional): Number of worker processes. Defaults to CPU count.
        calculation_dtype_precision (str, optional): Data type used for internal calculations. Defaults to "float32".

    Returns:
        List[str]: Paths to the globally adjusted output raster images.
    """

    print("Start global regression")

    _check_raster_requirements(input_image_paths, debug_mode)

    nodata_val = _get_nodata_value(input_image_paths, custom_nodata_value)

    if debug_mode: print("Calculating statistics")
    with rasterio.open(input_image_paths[0]) as src: num_bands = src.count
    num_images = len(input_image_paths)

    all_bounds = {}
    for idx, p in enumerate(input_image_paths):
        with rasterio.open(p) as ds:
            all_bounds[idx] = ds.bounds

    overlapping_pairs = _find_overlaps(all_bounds)

    all_overlap_stats = {}
    for id_i, id_j in overlapping_pairs:
        stats = _calculate_overlap_stats(
            num_bands,
            input_image_paths[id_i],
            input_image_paths[id_j],
            id_i,
            id_j,
            all_bounds[id_i],
            all_bounds[id_j],
            nodata_val,
            nodata_val,
            vector_mask_path=vector_mask_path,
            window_size=window_size,
            debug_mode=debug_mode,
        )
        all_overlap_stats.update(
            {
                k_i: {
                    **all_overlap_stats.get(k_i, {}),
                    **{
                        k_j: {**all_overlap_stats.get(k_i, {}).get(k_j, {}), **s}
                        for k_j, s in v.items()
                    },
                }
                for k_i, v in stats.items()
            }
        )

    all_whole_stats = {}
    for idx, path in enumerate(input_image_paths):
        all_whole_stats.update(
            _calculate_whole_stats(
                input_image_path=path,
                nodata=nodata_val,
                num_bands=num_bands,
                image_id=idx,
                vector_mask_path=vector_mask_path,
                window_size=window_size,
            )
        )

    all_params = np.zeros((num_bands, 2 * num_images, 1), dtype=float)
    for b in range(num_bands):
        if debug_mode: print(f"Processing band {b} for {num_images} images")

        A, y, tot_overlap = [], [], 0
        for i in range(num_images):

            for j in range(i + 1, num_images):
                stat = all_overlap_stats.get(i, {}).get(j)
                if stat is None:
                    continue
                s = stat[b]["size"]
                m1, v1 = stat[b]["mean"], stat[b]["std"]
                m2, v2 = (
                    all_overlap_stats[j][i][b]["mean"],
                    all_overlap_stats[j][i][b]["std"],
                )
                row_m = [0] * (2 * num_images)
                row_s = [0] * (2 * num_images)
                row_m[2 * i : 2 * i + 2] = [m1, 1]
                row_m[2 * j : 2 * j + 2] = [-m2, -1]
                row_s[2 * i], row_s[2 * j] = v1, -v2
                A.extend(
                    [
                        [v * s * custom_mean_factor for v in row_m],
                        [v * s * custom_std_factor for v in row_s],
                    ]
                )
                y.extend([0, 0])
                tot_overlap += s
        pjj = 1.0 if tot_overlap == 0 else tot_overlap / (2.0 * num_images)
        for j in range(num_images):
            mj = all_whole_stats[j][b]["mean"]
            vj = all_whole_stats[j][b]["std"]
            row_m = [0] * (2 * num_images)
            row_s = [0] * (2 * num_images)
            row_m[2 * j : 2 * j + 2] = [mj * pjj, 1 * pjj]
            row_s[2 * j] = vj * pjj
            A.extend([row_m, row_s])
            y.extend([mj * pjj, vj * pjj])

        A_arr = np.asarray(A)
        y_arr = np.asarray(y)
        res = least_squares(lambda p: np.asarray(A) @ p - np.asarray(y), [1, 0] * num_images)
        if debug_mode:
            overlap_pairs = overlapping_pairs
            _print_constraint_system(
                constraint_matrix=A_arr,
                adjustment_params=res.x,
                observed_values_vector=y_arr,
                overlap_pairs=overlap_pairs,
                num_images=num_images,
            )

        all_params[b] = res.x.reshape((2 * num_images, 1))

    if not os.path.exists(output_image_folder): os.makedirs(output_image_folder)
    out_paths: List[str] = []

    if parallel and max_workers is None:
        max_workers = mp.cpu_count()

    for img_idx, img_path in enumerate(input_image_paths):
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_image_folder, f"{base}{output_global_basename}.tif")
        out_paths.append(str(out_path))

        if debug_mode: print(f"Apply adjustments and saving results for {base}")
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
                    a = all_params[b, 2 * img_idx, 0]
                    b0 = all_params[b, 2 * img_idx + 1, 0]

                    if parallel:
                        futs = [
                            pool.submit(_process_tile_global,
                                        w,
                                        b,
                                        a,
                                        b0,
                                        nodata_val,
                                        calculation_dtype_precision,
                                        debug_mode,
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
                                debug_mode,
                            )
                            dst.write(buf.astype(meta["dtype"]), b + 1, window=win)
                if parallel:
                    pool.shutdown()

    print("Finished global regression")
    return out_paths

def _process_tile_global(
    window: Window,
    band_idx: int,
    a: float,
    b: float,
    nodata: int | float,
    calculation_dtype_precision: str,
    debug_mode: bool,
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
        debug_mode (bool): If True, prints processing information.

    Returns:
        Tuple[Window, np.ndarray]: Window and the adjusted data block.
    """

    if debug_mode: print(f"Processing band: {band_idx}, window: {window}")
    ds = _worker_dataset_cache["ds"]
    block = ds.read(band_idx + 1, window=window).astype(calculation_dtype_precision)

    mask = block != nodata
    block[mask] = a * block[mask] + b
    return window, block


def _print_constraint_system(
    constraint_matrix: ndarray,
    adjustment_params: ndarray,
    observed_values_vector: ndarray,
    overlap_pairs: tuple,
    num_images: int,
    ):
    """
    Prints the constraint matrix system with labeled rows and columns for debugging regression inputs.

    Args:
        constraint_matrix (ndarray): Coefficient matrix used in the regression system.
        adjustment_params (ndarray): Solved adjustment parameters (regression output).
        observed_values_vector (ndarray): Target values in the regression system.
        overlap_pairs (tuple): Pairs of overlapping image indices used in constraints.
        num_images (int): Total number of images in the system.

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

    # Build row labels
    row_labels = []
    for i, j in overlap_pairs:
        row_labels.append(f"Overlap({i}-{j}) Mean Diff")
        row_labels.append(f"Overlap({i}-{j}) Std Diff")

    for img_idx in range(num_images):
        row_labels.append(f"Image {img_idx} Mean Cnstr")
        row_labels.append(f"Image {img_idx} Std Cnstr")

    # Build column labels
    col_labels = []
    for i in range(num_images):
        col_labels.append(f"a{i}")
        col_labels.append(f"b{i}")

    # Print column headers
    header = " " * 24  # extra space for row label
    for lbl in col_labels:
        header += f"{lbl:>18}"
    print(header)

    # Print matrix rows
    for row_label, row in zip(row_labels, constraint_matrix):
        line = f"{row_label:>24}"  # adjust the width
        for val in row:
            line += f"{val:18.3f}"
        print(line)

    print("\nadjustment_params:")
    np.savetxt(sys.stdout, adjustment_params, fmt="%18.3f")

    print("\nobserved_values_vector:")
    np.savetxt(sys.stdout, observed_values_vector, fmt="%18.3f")

def _find_overlaps(
    image_bounds_dict: dict,
    ):
    """
    Finds all pairs of images with overlapping spatial bounds.

    Args:
        image_bounds_dict (dict): Dictionary mapping image indices to their rasterio bounds.

    Returns:
        Tuple[Tuple[int, int], ...]: Pairs of image indices with overlapping extents.
    """

    overlaps = []

    for key1, bounds1 in image_bounds_dict.items():
        for key2, bounds2 in image_bounds_dict.items():
            if key1 < key2:  # Avoid duplicate and self-comparison
                if (
                    bounds1.left < bounds2.right
                    and bounds1.right > bounds2.left
                    and bounds1.bottom < bounds2.top
                    and bounds1.top > bounds2.bottom
                ):
                    overlaps.append((key1, key2))

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
    debug_mode: bool =False,
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
        debug_mode (bool, optional): If True, prints debug information. Defaults to False.

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

        if debug_mode: print(f"Overlap bounds: x: {x_min:.2f} to {x_max:.2f}, y: {y_min:.2f} to {y_max:.2f}")

        if x_min >= x_max or y_min >= y_max:
            return stats

        row_min_i, col_min_i = rowcol(transform_i, x_min, y_max)
        row_max_i, col_max_i = rowcol(transform_i, x_max, y_min)
        row_min_j, col_min_j = rowcol(transform_j, x_min, y_max)
        row_max_j, col_max_j = rowcol(transform_j, x_max, y_min)

        height = min(row_max_i - row_min_i, row_max_j - row_min_j)
        width = min(col_max_i - col_min_i, col_max_j - col_min_j)

        if debug_mode:
            print(f"For overlap {os.path.basename(input_image_path_i)} with {os.path.basename(input_image_path_j)}:")
            print(f" - Pixel window {os.path.basename(input_image_path_i)}: cols: {col_min_i} to {col_max_i} ({col_max_i - col_min_i}), rows: {row_min_i} to {row_max_i} ({row_max_i - row_min_i})")
            print(f" - Pixel window {os.path.basename(input_image_path_j)}: cols: {col_min_j} to {col_max_j} ({col_max_j - col_min_j}), rows: {row_min_j} to {row_max_j} ({row_max_j - row_min_j})")

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
    image_id: int,
    vector_mask_path: str=None,
    window_size: tuple = None,
    ):
    """
    Computes mean, standard deviation, and valid pixel count for each band in a single image.

    Args:
        input_image_path (str): Path to the input raster image.
        nodata (int | float): NoData value to ignore during calculations.
        num_bands (int): Number of bands to process.
        image_id (int): Unique ID for the image, used as a key in the output.
        vector_mask_path (str, optional): Optional vector mask path to restrict statistics to masked regions. Defaults to None.
        window_size (tuple, optional): Optional tile size for block-wise processing. Defaults to None.

    Returns:
        dict: Dictionary with image ID as key and per-band statistics as sub-dictionary.
    """

    stats = {image_id: {}}

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

            stats[image_id][band_idx] = {
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