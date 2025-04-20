import tempfile
import os
import rasterio

from rasterio.warp import aligned_target
from rasterio.warp import reproject
from rasterio.enums import Resampling
from rasterio.windows import Window
from typing import Literal

def create_windows(width, height, tile_width, tile_height):
    for row_off in range(0, height, tile_height):
        for col_off in range(0, width, tile_width):
            win_width = min(tile_width, width - col_off)
            win_height = min(tile_height, height - row_off)
            yield Window(col_off, row_off, win_width, win_height)

def _align_rasters(
    input_image_paths: list[str],
    resample_method: Literal["nearest", "bilinear", "cubic", "average", "mode", "max", "min", "med", "q1", "q3"] = "bilinear",
    tap: bool = True,
    ) -> list[str]:
    temp_dir = tempfile.mkdtemp()  # Persistent temp directory
    aligned_paths = []

    # 1. Determine highest resolution
    best_resolution = float("inf")
    for path in input_image_paths:
        with rasterio.open(path) as src:
            res = min(abs(src.transform.a), abs(src.transform.e))
            if res < best_resolution:
                best_resolution = res
    target_res = (best_resolution, best_resolution)

    # 2. Reproject each image to its own aligned grid using target resolution
    for path in input_image_paths:
        filename = os.path.basename(path)
        output_path = os.path.join(temp_dir, filename)

        with rasterio.open(path) as src:
            dst_crs = src.crs
            transform = src.transform
            width = src.width
            height = src.height

            if tap:
                dst_transform, dst_width, dst_height = aligned_target(
                    transform, width, height, target_res
                )
            else:
                dst_transform, dst_width, dst_height = transform, width, height

            profile = src.profile.copy()
            profile.update({
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
                "crs": dst_crs,
            })

            with rasterio.open(output_path, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=getattr(Resampling, resample_method),
                    )

        aligned_paths.append(output_path)

    return aligned_paths