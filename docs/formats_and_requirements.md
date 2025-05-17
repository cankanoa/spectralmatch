# File Formats and Input Requirements

## Input Raster Requirements
Input rasters must meet specific criteria to ensure compatibility during processing. These are checked by _check_raster_requirements():

- Have a valid geotransform
- Share the same coordinate reference system (CRS)
- Have an identical number of bands
- Use consistent nodata values

Additionally, all rasters should:

 - Be a `.tif` file
 - Have overlap which represents the same data in each raster
 - Have a consistent spectral profile 

## Regression Parameters File
Regression parameters can be stored in a `json` file which includes:

 - Adjustments: Per-band scale and offset values applied to each image.
 - Whole Stats: Per-band mean, std, and size representing overall image statistics.
 - Overlap Stats: Per-image pair mean, std, and size for overlapping geometry regions.

The structure is a dictionary keyed by images basenames (no extension) with the following format:

```json
{
  "image_name": {
    "adjustments": {
      "band_0": {"scale": float, "offset": float},
      ...
    },
    "whole_stats": {
      "band_0": {"mean": float, "std": float, "size": int},
      ...
    },
    "overlap_stats": {
      "other_image": {
        "band_0": {"mean": float, "std": float, "size": int},
        ...
      },
      ...
    }
  },
  ...
}
```
This format represents the following: For each image_name there are adjustment, whole_stats and overlap_stats. For each adjustments, for each band, there is scale and offset. For each whole_stats and overlap_stats, for each band, there is mean, std, and size (number of pixels). Each band key follows the format band_0, band_1, etc. Mean and std are floats and size is an integer.

This structure is validated by `_validate_adjustment_model_structure()` before use to ensure consistency and completeness across images and bands. Global regression does not actually use 'adjustments' field because they are recalculated every run.

## Block Maps File
Block maps are spatial summaries of raster data, where each block represents the mean values of a group of pixels over a fixed region. They are used to reduce image resolution while preserving local radiometric characteristics, enabling efficient comparison and adjustment across images. Each map is structured as a grid of blocks with values for each spectral band. They can be saved as regular `geotif` files and together store this information: block_local_means, block_reference_mean, num_row, num_col, bounds_canvas_coords. 

There are two types of block maps, although their format is exactly the same:

 - **Local Block Map:** Each block stores the mean value of all pixels within its boundary for a single image.
 - **Reference Block Map:** Each block is the mean of all images means for its boundary; simply the mean of all local block maps.

Both block maps have the shape: `num_row, num_col, num_bands`, however, there are multiple (one for each image) local block maps and only one reference block map. Once a reference block map is created it is unique to its input images and cannot be accurately modified to add additional images. However, images can be 'brought' to a reference block map even if they were not involved in its creation as long as it covers that image.
