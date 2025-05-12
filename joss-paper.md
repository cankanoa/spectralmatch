---
title: 'spectralmatch'
tags:
  - Relative Radiometric Normalization
  - Python
  - Time series
  - Mosaic
  - GIS
  - QGIS plugin
  - Remote sensing
  - Histogram matching
  - Seamline
authors:
  - name: Kanoa Lindiwe
    orcid: 0009-0009-5520-1911
    affiliation: 1
  - name: Joseph Emile Honour Percival
    orcid: 0000-0001-5941-4601
    affiliation: 1
  - name: Ryan Perroy
    orcid: 0000-0002-4210-3281
    affiliation: 1
affiliations:
 - name: University of Hawaii at Hilo, United States
   index: 1
   ror: 02mp2av58
date: 8 May 2025
bibliography: paper.bib
---
# Summary
This library provides open-source tools for Relative Radiometric Normalization (RRN) to improve the spectral consistency of satellite image mosaics and time series. Designed for geoscientific applications requiring spatial or temporal composites, it includes global regression and local block adjustment algorithms that minimize inter-image variability without relying on ancillary data. Additional utilities support cloud and vegetation masking, Pseudo Invariant Feature (PIF) extraction, seamline generation, raster merging, and figure creation. The tool is available as an open-source Python library and QGIS plugin.

# Statement of Need
Geoscientific fields use image mosaics and time series for analysis. Mosaics allow multiple small images to cover a larger area, thus increasing the spatial coverage. Time series allow multiple images to cover a larger time span, thus increasing the temporal coverage. However, both formats are affected by inter-image spectral variability, caused by sensor differences, atmospheric conditions, illumination conditions, surface conditions, acquisition geometry, atmospheric scattering, and adjacency effect. These errors introduce inconsistencies, reduce accuracy in image analysis, and complicate the detection of actual environmental changes. To address these issues, two radiometric correction methods have been explored: Absolute Radiometric Correction (ARC) and Relative Radiometric Normalization (RRC). The absolute approach compensates for errors using in-situ sensor, atmospheric, geometric, and solar measurements, which is often difficult or impossible, particularly for historic images. Even when done in controlled conditions, with the absolute method the error can be XXX. Conversely, the relative approach applies algorithms to minimize the spectral difference between images, often by applying a gain and offset to match the images' mean and standard deviation. These methods do not attempt to find the actual spectral values but instead match images for consistent analysis without the need for ancillary data. A well-developed, open-source method for relative radiometric normalization does not currently exist, which this library aims to address through a Python library and a QGIS plugin.

# Use Cases and Alternatives

ENVI

ARCGIS

QGIS Bag plugin

[GitHub - ArminMoghimi/LIRRN: LIRRN: Location-Independent Relative Radiometric Normalization of Bitemporal Remote-Sensing Images](https://github.com/ArminMoghimi/LIRRN/tree/main)

[GitHub - chlsl/rrn-multisensor-multidate: Implementation for ISPRS 2020 paper "Relative Radiometric Normalization Using Several Automatically Chosen Reference Images For Multi-Sensor, Multi-Temporal Series"](https://github.com/chlsl/rrn-multisensor-multidate?tab=readme-ov-file)

[GitHub - SMByC/ArrNorm: Automatic Relative Radiometric Normalization](https://github.com/SMByC/ArrNorm)


# Algorithms
The global regression feature performs radiometric normalization across overlapping satellite images by aligning their brightness and contrast globally. It first detects overlapping image pairs and computes per-band statistics (mean, standard deviation) within those regions. Using these statistics, a least-squares regression system is constructed to solve for per-image, per-band scale and offset parameters that minimize radiometric differences in overlaps. This approach aims to minimize brightness and contrast differences across images while preserving global consistency and aligning the spectral profiles of images to a central tendency. The method supports nodata-aware processing for images of irregular shapes and internal gaps, vector Pseudo Invariant Feature masking, custom-weighted mean and standard deviation constraints, efficient windowing and parallelization for large datasets and cloud processing, and progress saving and resumption enabled by stored statistics.

The local block adjustment feature applies block-wise radiometric correction to individual satellite images based on local differences from a reference mosaic. The method divides the combined extent of all input images into spatial blocks and calculates local mean statistics for each block. Each image is then locally adjusted using interpolated adaptive gamma normalization to align with the global reference mosaic. This allows radiometric consistency across spatially heterogeneous scenes on a block scale. The method supports nodata-aware processing for images of irregular shapes and internal gaps, vector Pseudo Invariant Feature masking, multiple strategies for determining block layout, efficient windowing and parallelization for large datasets and cloud processing, and progress saving and resumption enabled by stored block maps.

Various helper functions support the creation of cloud masks, non-vegetation PIFs, generating seamlines, merging images, and basic figures. Cloud masking utilities enable the generation of binary masks using OmniCloudMask, followed by post-processing and vectorization. Vegetation masking utilities use NDVI-based thresholds, followed by post-processing and vectorization. The created masks can be manually modified, if desired, and used to mask input images or withhold pixels from analysis. Seamline generation utilities use Voronoi-based centerlines to define mosaic divisions and produce edge-aware boundaries. Statistical utilities can generate basic figures comparing image spectral profiles before and after matching to evaluate radiometric changes. Raster merging utilities combine the final images into a seamless mosaic.
# Figures
![Spectral Comparision](https://raw.githubusercontent.com/spectralmatch/spectralmatch/main/images/matching_histogram.png)
_Figure 1. Shows the average spectral profile of two WorldView 3 images before and after global to local matching._
# References

Yu, L., Zhang, Y., Sun, M., Zhou, X., & Liu, C. (2017). An auto-adapting global-to-local color balancing method for optical imagery mosaic. ISPRS Journal of Photogrammetry and Remote Sensing, 132, 1–19. https://doi.org/10.1016/j.isprsjprs.2017.08.002

Yuan, X., Cai, Y., & Yuan, W. (2023). Voronoi Centerline-Based Seamline Network Generation Method. Remote Sensing, 15(4), Article 4. https://doi.org/10.3390/rs15040917

Wright, N., Duncan, J. M. A., Callow, J. N., Thompson, S. E., & George, R. J. (2025). Training sensor-agnostic deep learning models for remote sensing: Achieving state-of-the-art cloud and cloud shadow identification with OmniCloudMask. Remote Sensing of Environment, 322, 114694. https://doi.org/10.1016/j.rse.2025.114694
