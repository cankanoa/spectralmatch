import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple, Literal
import os

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.optimize import least_squares

from spectralmatch.utils.utils_local import (
    _apply_gamma_correction,
    _compute_block_size,
    _compute_blocks,
    _download_block_map,
    _get_bounding_rectangle,
    _weighted_bilinear_interpolation,
)
from spectralmatch.utils.utils_common import (
    _check_raster_requirements,
    _create_windows,
    _get_nodata_value,
)
from spectralmatch.utils.utils_global import (
    _calculate_overlap_stats,
    _calculate_whole_stats,
    _find_overlaps,
    _print_constraint_system,
)

_worker_ds = None
_worker_dtype = None


def _choose_context(
    prefer_fork: bool = True
    ) -> mp.context.BaseContext:

    """
    Chooses and returns the most suitable multiprocessing context based on the given
    preference and the operating system.

    This function attempts to decide the multiprocessing context based on whether
    the platform supports forking or not and user preference. Fork contexts are given
    priority on Linux systems. For macOS, it tries to utilize a fork context, but if
    unsupported, falls back to other options. On other platforms, it defaults to
    "forkserver" or "spawn" if no other option is available.

    Args:
    prefer_fork (bool): A boolean flag indicating whether to prioritize the
    "fork" context when available. Defaults to True.

    Returns:
    mp.context.BaseContext: The multiprocessing context selected based
    on the provided preference and platform compatibility.
    """

    if prefer_fork and sys.platform.startswith("linux"):
        return mp.get_context("fork")
    if prefer_fork and sys.platform == "darwin":
        try:
            return mp.get_context("fork")
        except ValueError:
            pass
    try:
        return mp.get_context("forkserver")
    except ValueError:
        return mp.get_context("spawn")


def _init_worker(
    img_path: str,
    calc_dtype: str
    ):

    """
    Initializes a worker for raster data processing.

    This function sets up a worker with the specified image path and calculation
    data type. It opens the raster data using the provided path and stores the
    specified calculation data type for future reference by the worker processes.

    Args:
    img_path (str): Path to the raster image file to be processed.
    calc_dtype (str): Calculation data type to be used by the worker.
    """

    global _worker_ds, _worker_dtype
    _worker_ds = rasterio.open(img_path, "r")
    _worker_dtype = calc_dtype


def _process_tile_global(
    window: Window,
    band_idx: int,
    a: float,
    b: float,
    nodata,
    debug_mode: bool,
    ):

    """
    Processes a specific global tile within a dataset by applying a linear transformation
    to the pixel values based on the provided coefficients while handling no-data values.

    Args:
    window (Window): The spatial window of the tile to be processed, representing a
    portion of the dataset.
    band_idx (int): The index of the band within the dataset to be processed.
    a (float): The multiplier coefficient for the linear transformation.
    b (float): The additive coefficient for the linear transformation.
    nodata: The value representing no-data in the dataset. Pixels with this value
    will not undergo processing.

    Returns:
    tuple[Window, numpy.ndarray]: A tuple where the first element is the processed
    spatial window, and the second element is the transformed array of pixel values
    for the given tile.
    """

    if debug_mode: print(f"Processing band: {band_idx}, window: {window}")
    block = _worker_ds.read(band_idx + 1, window=window).astype(_worker_dtype)
    mask = block != nodata
    block[mask] = a * block[mask] + b
    return window, block


def _compute_tile_local(
    window: Window,
    band_idx: int,
    M,
    N,
    bounding_rect,
    block_ref_mean,
    block_loc_mean,
    nodata_val,
    alpha,
    correction_method,
    calculation_dtype_precision,
    debug_mode,
    ):

    if debug_mode: print(f"Processing band: {band_idx}")

    arr_in = _worker_ds.read(band_idx + 1, window=window).astype(calculation_dtype_precision)
    arr_out = np.full_like(arr_in, nodata_val, dtype=calculation_dtype_precision)

    mask = arr_in != nodata_val
    if not np.any(mask):
        return window, band_idx, arr_out

    vr, vc = np.where(mask)

    win_tr = _worker_ds.window_transform(window)
    col_coords = win_tr[2] + np.arange(window.width) * win_tr[0]
    row_coords = win_tr[5] + np.arange(window.height) * win_tr[4]

    row_f = np.clip(
        ((bounding_rect[3] - row_coords) / (bounding_rect[3] - bounding_rect[1])) * M
        - 0.5,
        0,
        M - 1,
    )
    col_f = np.clip(
        ((col_coords - bounding_rect[0]) / (bounding_rect[2] - bounding_rect[0])) * N
        - 0.5,
        0,
        N - 1,
    )

    ref = _weighted_bilinear_interpolation(
        block_ref_mean[:, :, band_idx], col_f[vc], row_f[vr]
    )
    loc = _weighted_bilinear_interpolation(
        block_loc_mean[:, :, band_idx], col_f[vc], row_f[vr]
    )
    # if debug_mode:
    #     gammas_array = np.full(arr_in.shape, global_nodata_value, dtype=calculation_dtype_precision)
    #     gammas_array[valid_rows, valid_cols] = gammas
    #     _download_block_map(
    #         block_map=gammas_array,
    #         bounding_rect=this_image_bounds,
    #         output_image_path=os.path.join(output_image_folder, "Gamma", out_name + f"_Gamma.tif"),
    #         projection=projection,
    #         nodata_value=global_nodata_value,
    #         output_bands_map=(b+1,),
    #         override_band_count=num_bands,
    #     )

    # if debug_mode:
    #     _download_block_map(
    #         block_map=np.where(valid_mask, 1, global_nodata_value),
    #         bounding_rect=this_image_bounds,
    #         output_image_path=os.path.join(output_image_folder, "ValidMasks", out_name + f"_ValidMask.tif"),
    #         projection=projection,
    #         nodata_value=global_nodata_value,
    #         output_bands_map=(b+1,),
    #         override_band_count=num_bands
    #     )

    # if debug_mode:
    #     _download_block_map(
    #         block_map=reference_band,
    #         bounding_rect=this_image_bounds,
    #         output_image_path=os.path.join(output_image_folder, "ReferenceBand", out_name + f"_ReferenceBand.tif"),
    #         projection=projection,
    #         nodata_value=global_nodata_value,
    #         output_bands_map=(b+1,),
    #         override_band_count=num_bands,
    #     )
    #
    # if debug_mode:
    #     _download_block_map(
    #         block_map=local_band,
    #         bounding_rect=this_image_bounds,
    #         output_image_path=os.path.join(output_image_folder, "LocalBand", out_name + f"_LocalBand.tif"),
    #         projection=projection,
    #         nodata_value=global_nodata_value,
    #         output_bands_map=(b+1,),
    #         override_band_count=num_bands,
    #     )

    if correction_method == "gamma":
        smallest = np.min([arr_in[mask], ref, loc])
        if smallest <= 0:
            offset = abs(smallest) + 1
            arr_out[mask], _ = _apply_gamma_correction(
                arr_in[mask] + offset,
                ref + offset,
                loc + offset,
                alpha,
            )
            arr_out[mask] -= offset
        else:
            arr_out[mask], _ = _apply_gamma_correction(arr_in[mask], ref, loc, alpha)
    elif correction_method == "linear":
        arr_out[mask] = arr_in[mask] * (ref / loc)
    else: raise ValueError('Invalid correction method')

    return window, band_idx, arr_out


def global_match(
    input_image_paths: List[str],
    output_image_folder: str,
    *,
    custom_mean_factor: float = 1.0,
    custom_std_factor: float = 1.0,
    output_global_basename: str = "_global",
    vector_mask_path: Optional[str] = None,
    tile_width_and_height_tuple: Optional[Tuple[int, int]] = None,
    debug_mode: bool = False,
    custom_nodata_value: float | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
    calc_dtype: str = "float32",
    ):

    """
    Matches multiple input raster images to a common global statistical range using overlapping areas
    for deriving statistical adjustments. Adjustments include mean and standard deviation factors, ensuring
    the images are globally consistent in derived values.

    This function processes pairs of overlapping raster images to compute statistical parameters, combines
    global statistics for the entire collection of raster images, and generates output raster images with
    adjusted pixel values. Additional features such as custom nodata values, vector mask usage, and tiling
    are supported. The function allows parallel processing to optimize performance.

    Args:
    input_image_paths (List[str]): Paths to input raster images.
    output_image_folder (str): Directory where adjusted images will be saved.
    custom_mean_factor (float): Scale factor for mean adjustment, default is 1.0.
    custom_std_factor (float): Scale factor for standard deviation adjustment, default is 1.0.
    output_global_basename (str): Global basename suffix added to output image filenames, default is "_global".
    vector_mask_path (Optional[str]): Path to a vector mask file used to limit processing, optional.
    tile_width_and_height_tuple (Optional[Tuple[int, int]]): Tuple specifying tile width and height for tiled processing, optional.
    debug_mode (bool): Enables debug mode when set to True, default is False.
    custom_nodata_value (float | None): Override nodata value for rasters, optional.
    parallel (bool): Enables parallel processing when set to True, default is False.
    max_workers (int | None): Maximum number of workers for parallel processing, default is None.
    calc_dtype (str): Data type used for intermediate calculations, default is "float32".

    Returns:
    List[str]: Paths to the output adjusted raster images.
    """
    print("Start global matching")

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
            tile_width_and_height_tuple=tile_width_and_height_tuple,
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
                tile_width_and_height_tuple=tile_width_and_height_tuple,
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

    img_dir = os.path.join(output_image_folder, "Images")
    if not os.path.exists(img_dir): os.makedirs(img_dir)
    out_paths: List[str] = []

    if parallel and max_workers is None:
        max_workers = mp.cpu_count()

    for img_idx, img_path in enumerate(input_image_paths):
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(img_dir, f"{base}{output_global_basename}.tif")
        out_paths.append(str(out_path))

        if debug_mode: print(f"Apply adjustments and saving results for {base}")
        with rasterio.open(img_path) as src:
            meta = src.meta.copy()
            meta.update({"count": num_bands, "nodata": nodata_val})
            with rasterio.open(out_path, "w", **meta) as dst:

                if tile_width_and_height_tuple:
                    tw, th = tile_width_and_height_tuple
                    windows = list(_create_windows(src.width, src.height, tw, th))
                else:
                    windows = [Window(0, 0, src.width, src.height)]

                if parallel:
                    ctx = _choose_context(prefer_fork=True)
                    pool = ProcessPoolExecutor(
                        max_workers=max_workers,
                        mp_context=ctx,
                        initializer=_init_worker,
                        initargs=(img_path, calc_dtype),
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

    print("Finished global matching")
    return out_paths


def local_match(
    input_image_paths: List[str],
    output_image_folder: str,
    *,
    output_local_basename: str = "_local",
    custom_nodata_value: float | None = None,
    target_blocks_per_image: int = 100,
    alpha: float = 1.0,
    calculation_dtype_precision: str = "float32",
    output_dtype: str = "float32",
    projection: str = "EPSG:4326",
    debug_mode: bool = False,
    tile_width_and_height_tuple: Optional[Tuple[int, int]] = None,
    correction_method: Literal["gamma", "linear"] = "gamma",
    parallel: bool = False,
    max_workers: int | None = None,
    ):

    """
    Matches histograms of input raster images using local histogram matching approach.
    This function processes raster images, adjusts their histograms locally based on
    reference blocks, and saves the corrected images to the specified output directory.
    The procedure operates block-wise on the raster images for local corrections and
    supports parallel execution for performance optimization.

    Args:
    input_image_paths (List[str]): A list of paths to input raster image files.
    output_image_folder (str): Directory path where the output images will be saved.
    output_local_basename (str): Suffix for the output filenames indicating local
    histogram matching, default is "_local".
    custom_nodata_value (float | None): Custom value to represent no data areas;
    if None, it is auto-detected from input rasters.
    target_blocks_per_image (int): Approximate number of blocks to divide each
    raster for local histogram matching, default is 100.
    alpha (float): Scaling factor for adjustment when applying histogram corrections,
    default is 1.0 (no scaling).
    calculation_dtype_precision (str): The data type precision used internally
    for corrections, default is "float32".
    output_dtype (str): Data type of the output images, default is "float32".
    projection (str): Coordinate reference system for the output rasters,
    default is "EPSG:4326".
    debug_mode (bool): Flag to enable saving intermediate block map for debugging,
    default is False.
    tile_width_and_height_tuple (Optional[Tuple[int, int]]): Optional tuple
    specifying the width and height of tiles for processing input raster,
    default is None.
    correction_method (Literal["gamma", "linear"]): Method for histogram correction,
    either "gamma" or "linear." Default is "gamma".
    parallel (bool): If True, enables parallel processing for higher efficiency,
    default is False.
    max_workers (int | None): Limits the number of parallel workers, default is
    None (uses system CPU count if parallel is True).

    Returns:
    List[str]: List of file paths to the output raster images that have been
    locally histogram-matched.
    """
    print("Start local matching")
    _check_raster_requirements(input_image_paths, debug_mode)

    nodata_val = _get_nodata_value(input_image_paths, custom_nodata_value)
    if debug_mode: print(f"Global nodata value: {nodata_val}")

    out_img_dir = os.path.join(output_image_folder, "Images")
    if not os.path.exists(out_img_dir): os.makedirs(out_img_dir)

    bounding_rect = _get_bounding_rectangle(input_image_paths)
    M, N = _compute_block_size(input_image_paths, target_blocks_per_image, bounding_rect)
    # M, N = _compute_mosaic_coefficient_of_variation(input_image_paths, global_nodata_value) # Aproach from the paper to compute bock size

    with rasterio.open(input_image_paths[0]) as ds:
        num_bands = ds.count

    if debug_mode: print("Computing global reference block map")
    block_ref_mean, _ = _compute_blocks(
        input_image_paths,
        bounding_rect,
        M,
        N,
        num_bands,
        nodata_value=nodata_val,
        tile_width_and_height_tuple=tile_width_and_height_tuple,
    )

    if debug_mode:
        _download_block_map(
            block_map=np.nan_to_num(block_ref_mean, nan=nodata_val),
            bounding_rect=bounding_rect,
            output_image_path= os.path.join(output_image_folder, "BlockReferenceMean", "BlockReferenceMean.tif"),
            nodata_value=nodata_val,
            projection=projection,
        )

    if parallel and max_workers is None:
        max_workers = mp.cpu_count()

    out_paths: List[str] = []
    for img_path in input_image_paths:
        in_name = os.path.splitext(os.path.basename(img_path))[0]
        out_name = os.path.splitext(os.path.basename(img_path))[0] + output_local_basename
        out_path = os.path.join(out_img_dir, f"{out_name}.tif")
        out_paths.append(str(out_path))

        if debug_mode: print(f"Processing {in_name}")
        if debug_mode: print(f"Computing local block map")
        block_loc_mean, block_loc_count = _compute_blocks(
            [img_path],
            bounding_rect,
            M,
            N,
            num_bands,
            nodata_value=nodata_val,
            tile_width_and_height_tuple=tile_width_and_height_tuple,
        )

        # block_local_mean = _smooth_array(block_local_mean, nodata_value=global_nodata_value)

        if debug_mode:
            _download_block_map(
                block_map=np.nan_to_num(block_loc_mean, nan=nodata_val),
                bounding_rect=bounding_rect,
                output_image_path=os.path.join(output_image_folder, "BlockLocalMean", f"{out_name}_BlockLocalMean.tif"),
                nodata_value=nodata_val,
                projection=projection,
            )
            _download_block_map(
                block_map=np.nan_to_num(block_loc_count, nan=nodata_val),
                bounding_rect=bounding_rect,
                output_image_path=os.path.join(output_image_folder, "BlockLocalCount", f"{out_name}_BlockLocalCount.tif"),
                nodata_value=nodata_val,
                projection=projection,
            )

        if debug_mode: print(f"Computing local correction, applying, and saving")
        with rasterio.open(img_path) as src:
            meta = src.meta.copy()
            meta.update({"count": num_bands, "dtype": output_dtype, "nodata": nodata_val})
            with rasterio.open(out_path, "w", **meta) as dst:

                if tile_width_and_height_tuple:
                    tw, th = tile_width_and_height_tuple
                    windows = list(_create_windows(src.width, src.height, tw, th))
                else:
                    windows = [Window(0, 0, src.width, src.height)]

                if parallel:
                    ctx = _choose_context(prefer_fork=True)

                    pool = ProcessPoolExecutor(
                        max_workers=max_workers,
                        mp_context=ctx,
                        initializer=_init_worker,
                        initargs=(img_path, calculation_dtype_precision),
                    )

                    futures = [
                        pool.submit(_compute_tile_local,
                            w,
                            b,
                            M,
                            N,
                            bounding_rect,
                            block_ref_mean,
                            block_loc_mean,
                            nodata_val,
                            alpha,
                            correction_method,
                            calculation_dtype_precision,
                            debug_mode,
                            )
                        for b in range(num_bands)
                        for w in windows
                    ]
                    for fut in as_completed(futures):
                        win, b_idx, buf = fut.result()
                        dst.write(buf.astype(output_dtype), b_idx + 1, window=win)
                    pool.shutdown()
                else:
                    _init_worker(img_path, calculation_dtype_precision)

                    for b in range(num_bands):
                        for win in windows:
                            win_, b_idx, buf = _compute_tile_local(
                                win,
                                b,
                                M,
                                N,
                                bounding_rect,
                                block_ref_mean,
                                block_loc_mean,
                                nodata_val,
                                alpha,
                                correction_method,
                                calculation_dtype_precision,
                                debug_mode
                                )
                            dst.write(buf.astype(output_dtype), b_idx + 1, window=win_)
    print("Finished local matching")
    return out_paths
