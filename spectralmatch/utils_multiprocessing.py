import multiprocessing as mp
import os
import sys
import rasterio
import numpy as np

from rasterio.windows import Window
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Tuple, Literal, List, Callable, Any, Optional
from multiprocessing import shared_memory


def _choose_context(prefer_fork: bool = True) -> mp.context.BaseContext:

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


def _resolve_parallel_config(
    config: Tuple[Literal["process", "thread"], Literal["cpu"] | int] | None
) -> Tuple[bool, Optional[str], Optional[int]]:
    if config is None:
        return False, None, None
    backend, workers = config
    max_workers = os.cpu_count() if workers == "cpu" else int(workers)
    return True, backend, max_workers


def _get_executor(backend: str, max_workers: int, initializer=None, initargs=None):
    if backend == "process":
        return ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=initializer,
            initargs=initargs or ()
        )
    elif backend == "thread":
        return ThreadPoolExecutor(max_workers=max_workers)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _run_parallel_images(
    image_paths: List[str],
    run_parallel_windows: Callable[[str, Tuple], None],
    image_parallel_workers: Tuple[Literal["process", "thread"], Literal["cpu"] | int] | None = None,
    window_parallel_workers: Tuple[Literal["process", "thread"], Literal["cpu"] | int] | None = None,
):
    parallel, image_backend, image_max_workers = _resolve_parallel_config(image_parallel_workers)

    if parallel:
        with _get_executor(image_backend, image_max_workers) as image_pool:
            futures = [
                image_pool.submit(run_parallel_windows, path, window_parallel_workers)
                for path in image_paths
            ]
            for f in as_completed(futures):
                f.result()
    else:
        for path in image_paths:
            run_parallel_windows(path, window_parallel_workers)


def _run_parallel_windows(
    windows: List[Any],
    process_fn: Callable[[Any], Any],
    window_parallel_workers: Tuple[Literal["process", "thread"], Literal["cpu"] | int] | None = None,
):
    parallel, backend, max_workers = _resolve_parallel_config(window_parallel_workers)

    if parallel:
        with _get_executor(backend, max_workers) as executor:
            futures = [executor.submit(process_fn, win) for win in windows]
            for f in as_completed(futures):
                f.result()
    else:
        for win in windows:
            process_fn(win)


class WorkerContext:
    cache = {}

    @classmethod
    def init(cls, config: dict):
        """
        Initializes per-process context from a typed config dictionary.

        Each entry maps a key to a tuple describing how to initialize a resource:

            - ('raster', filepath): Open raster with rasterio.
            - ('shm', shm_name): Attach to shared memory.
            - ('array', shm_name, shape, dtype): Create NumPy array from shared memory.
            - ('value', literal): Store a direct Python value.

        Examples:
            {
                "input": ("raster", "/path/to/image.tif"),
                "weights": ("array", "shm_weights", (512, 512), "float32"),
                "debug": ("value", True)
            }

        Resources are stored in WorkerContext.cache and accessed via WorkerContext.get(key).
        """

        cls.cache = {}

        for key, value in config.items():
            if not isinstance(value, tuple) or not value:
                raise ValueError(f"Invalid config for key '{key}': must be a tuple")

            kind = value[0]
            if kind == "raster":
                cls.cache[key] = rasterio.open(value[1], "r")
            elif kind == "shm":
                cls.cache[key] = shared_memory.SharedMemory(name=value[1])
            elif kind == "array":
                _, shm_name, shape, dtype_name = value
                shm = shared_memory.SharedMemory(name=shm_name)
                arr = np.ndarray(shape, dtype=np.dtype(dtype_name), buffer=shm.buf)
                cls.cache[key] = arr
                cls.cache[f"{key}_shm"] = shm
            elif kind == "value":
                cls.cache[key] = value[1]
            else:
                raise ValueError(f"Unknown resource type '{kind}' for key '{key}'")

    @classmethod
    def get(cls, key):
        return cls.cache.get(key)

    @classmethod
    def close(cls):
        for key, obj in cls.cache.items():
            if hasattr(obj, "close"):
                obj.close()
        cls.cache.clear()


def _resolve_windows(
    dataset,
    window_size: int | Tuple[int, int] | Literal["internal"] | None,
    ) -> List[Window]:
    """
    Generates a list of windows based on the specified tiling strategy.

    Args:
        dataset (rasterio.DatasetReader): Open raster dataset.
        window_size (int | Tuple[int, int] | Literal["internal"] | None):
            Tiling strategy to use:
            - int: square tile size,
            - (int, int): custom tile width and height,
            - "internal": uses native tiling of the input raster,
            - None: single window covering the full image.

    Returns:
        List[Window]: A list of rasterio Windows covering the dataset.
    """
    width = dataset.width
    height = dataset.height

    if window_size == "internal":
        windows = [win for _, win in dataset.block_windows(1)]
    elif isinstance(window_size, int):
        windows = _create_windows(width, height, window_size, window_size)
    elif isinstance(window_size, tuple):
        windows = _create_windows(width, height, window_size[0], window_size[1])
    else:
        windows = [Window(0, 0, width, height)]

    return windows


def _create_windows(
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
    ):
    """
    Generates tiled windows across a raster based on specified dimensions.

    Args:
        width (int): Total width of the raster.
        height (int): Total height of the raster.
        tile_width (int): Width of each tile.
        tile_height (int): Height of each tile.

    Yields:
        rasterio.windows.Window: A window representing a tile's position and size.
    """

    for row_off in range(0, height, tile_height):
        for col_off in range(0, width, tile_width):
            win_width = min(tile_width, width - col_off)
            win_height = min(tile_height, height - row_off)
            yield Window(col_off, row_off, win_width, win_height)
