import os
import tempfile
from osgeo import gdal

def merge_images_with_masks(input_dictionary_image_then_mask, output_merge_path,
                            apply_tap=True, resample_resolution="highest",
                            resample_method="bilinear"):
    """
Aligns images before masking, applies vector masks using GDAL Warp, and merges the final output.

:param input_dictionary_image_then_mask: Dictionary with image paths as keys and vector mask paths as values.
:param output_merge_path: Path to save the merged image.
:param apply_tap: Whether to apply Target Aligned Pixels (TAP) before masking.
:param resample_resolution: How to determine the resolution (options: "highest", "lowest", "average").
:param resample_method: GDAL resampling method as a string (e.g., "bilinear", "cubic").
"""

    # Create output directories
    masked_folder = os.path.join(output_merge_path, "MaskedImages")
    os.makedirs(output_merge_path, exist_ok=True)
    os.makedirs(masked_folder, exist_ok=True)

    masked_images = []
    resolutions = []

    # Step 1: Gather resolutions from all images
    for image_path in input_dictionary_image_then_mask.keys():
        image_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
        if image_ds is None:
            print(f"Error: Could not open image {image_path}")
            continue

        xRes = abs(image_ds.GetGeoTransform()[1])  # Pixel width
        yRes = abs(image_ds.GetGeoTransform()[5])  # Pixel height
        resolutions.append((xRes, yRes))
        image_ds = None  # Close dataset

    if not resolutions:
        print("No valid images found. Exiting.")
        return

    # Step 2: Determine the target resolution
    if resample_resolution == "highest":
        target_xRes = min(res[0] for res in resolutions)
        target_yRes = min(res[1] for res in resolutions)
    elif resample_resolution == "lowest":
        target_xRes = max(res[0] for res in resolutions)
        target_yRes = max(res[1] for res in resolutions)
    elif resample_resolution == "average":
        target_xRes = sum(res[0] for res in resolutions) / len(resolutions)
        target_yRes = sum(res[1] for res in resolutions) / len(resolutions)
    else:
        raise ValueError(f"Invalid resample_resolution value: {resample_resolution}")

    print(f"Using target resolution: xRes={target_xRes}, yRes={target_yRes}")

    # Step 3: Process each image
    for image_path, vector_mask_path in input_dictionary_image_then_mask.items():
        image_basename = os.path.basename(image_path)

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_tif:
            temp_tif_path = temp_tif.name

        # If apply_tap is enabled, resample before masking
        if apply_tap:
            gdal.Warp(temp_tif_path, image_path,
                      xRes=target_xRes, yRes=target_yRes,  # Apply determined resolution
                      targetAlignedPixels=True,  # Align pixels
                      resampleAlg=resample_method,  # Use resample method as a string
                      format="GTiff")
        else:
            temp_tif_path = image_path  # Use original image without resampling

        # Open aligned image to get NoData value
        aligned_ds = gdal.Open(temp_tif_path, gdal.GA_ReadOnly)
        if aligned_ds is None:
            print(f"Error: Could not align {image_path}")
            continue
        nodata_value = aligned_ds.GetRasterBand(1).GetNoDataValue()
        aligned_ds = None  # Close dataset

        # Create masked output path
        masked_output = os.path.join(masked_folder, f"{os.path.splitext(image_basename)[0]}_masked.tif")

        # Apply vector mask using GDAL Warp
        gdal.Warp(masked_output, temp_tif_path,
                  cutlineDSName=vector_mask_path,  # Use vector file as cutline
                  cropToCutline=True,  # Crops to the mask extent
                  warpOptions=["CUTLINE_ALL_TOUCHED=TRUE"],
                  dstNodata=nodata_value,  # Uses aligned image's NoData value
                  format="GTiff")

        if os.path.exists(masked_output):
            masked_images.append(masked_output)
        else:
            print(f"Error: Masking failed for {image_path}")

        # Delete temp file if TAP was applied
        if apply_tap and os.path.exists(temp_tif_path):
            os.remove(temp_tif_path)

    if not masked_images:
        print("No valid masked images found. Exiting.")
        return

    # Define merged output file path
    merged_output_path = os.path.join(output_merge_path, "merged_output.tif")

    # Merge all masked images using GDAL Warp
    gdal.Warp(merged_output_path, masked_images, format="GTiff", resampleAlg=resample_method)

    print(f"Masked images saved in: {masked_folder}")
    print(f"Successfully merged images into {merged_output_path}")

input_dict = {
    "/mnt/e/kanoa/LocalMatch/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFromUSGSLidar_Pansharps_GlobalMatch_LocalMatch.tif":
        "/mnt/e/kanoa/LocalMatch/Masks/Right.gpkg",
    "/mnt/e/kanoa/LocalMatch/17DEC08211800-M1BS-016445319010_01_P004_FLAASH_OrthoFromUSGSLidar_Pansharps_GlobalMatch_LocalMatch.tif":
        "/mnt/e/kanoa/LocalMatch/Masks/Right.gpkg",
    "/mnt/e/kanoa/LocalMatch/17DEC08211801-M1BS-016445319010_01_P005_FLAASH_OrthoFromUSGSLidar_Pansharps_GlobalMatch_LocalMatch.tif":
        "/mnt/e/kanoa/LocalMatch/Masks/Right.gpkg",
    "/mnt/e/kanoa/LocalMatch/17DEC08211840-M1BS-016445318010_01_P015_FLAASH_OrthoFromUSGSLidar_Pansharps_GlobalMatch_LocalMatch.tif":
        "/mnt/e/kanoa/LocalMatch/Masks/Left.gpkg",
    "/mnt/e/kanoa/LocalMatch/17DEC08211841-M1BS-016445318010_01_P016_FLAASH_OrthoFromUSGSLidar_Pansharps_GlobalMatch_LocalMatch.tif":
        "/mnt/e/kanoa/LocalMatch/Masks/Left.gpkg"
}

output_merge_path = "/mnt/e/kanoa/LocalMatch/MergedOutput"

merge_images_with_masks(input_dict, output_merge_path)