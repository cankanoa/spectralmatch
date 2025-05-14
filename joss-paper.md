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
---
# Summary  
Spectralmatch provides tools for Relative Radiometric Normalization (RRN) to enhance spectral consistency across image mosaics and time series. It is built for geoscientific use, with a sensor- and unit-agnostic design, optimized for automation and efficiency on large datasets. It includes global regression and local block adjustment algorithms that minimize inter-image variability without relying on ancillary data. The methodology is primarily derived from Yu et al. (2017). Additional utilities support cloud and vegetation masking, Pseudo Invariant Feature (PIF) based exclusion, seamline generation, raster merging, and figure creation. The tool is available as an open-source Python library and QGIS plugin.  
  
# Statement of Need  
Geoscientific fields use image mosaics and time series for analysis. Mosaics increase spatial coverage by combining images across areas, while time series extend temporal coverage by linking images across time. However, both formats are affected by inter-image spectral variability, caused by sensor differences, atmospheric conditions, illumination conditions, surface conditions, acquisition geometry, atmospheric scattering, and adjacency effect (Chavez, 1996). These errors introduce inconsistencies, reduce accuracy in image analysis, and complicate the detection of actual environmental changes. To address these issues, two correction methods have been explored: Absolute Radiometric Correction (ARC) and Relative Radiometric Normalization (RRN). The absolute approach compensates for errors using in-situ measurements, which is often difficult or impossible, particularly for historic images. Conversely, the relative approach applies algorithms to minimize the spectral difference between images, often by applying a gain and offset to match the images' mean and standard deviation. These methods do not attempt to find the actual spectral values but instead match images for consistent analysis without the need for ancillary data. This library addresses the current lack of a robust open-source method for relative radiometric normalization by providing a Python library and a QGIS plugin.
  
# Use Cases and Alternatives  
Researchers have examined various methods for performing RRN. Vorovencii & D.M. (2014) compared five time series normalization methods—histogram matching (HM), simple regression (SR), pseudo-invariant features (PIF), dark and bright set (DB), and a no-change set derived from scattergrams (NC)—and found HM, SR, and NC to perform best. Canty and Nielsen (2008) applied Iteratively Re-weighted Multivariate Alteration Detection (IR-MAD) using custom IDL scripts on Landsat imagery. Gan et al. (2021) utilized MODIS reference data to enhance the consistency of Landsat and Sentinel imagery, using custom code in MATLAB. While various methods have been studied, most are not integrated into software packages, and their algorithms are typically not publicly released. This leaves researchers either spending significant time implementing their own versions of the algorithms or relying on the limited available tools. QGIS includes basic histogram matching and limited IR-MAD through the [Histogram Matching](https://github.com/Gustavoohs/HistMatch) and [ArrNorm](https://github.com/SMByC/ArrNorm) plugins, respectively. ArrNorm also provides a command-line interface. ArcGIS Pro offers multiple color balancing algorithms within its ecosystem, which are dodging, global fit, histogram, and standard deviation. ENVI offers basic histogram matching algorithms based on a single image overlapping area or entirety. In addition to these software, some researchers have shared their code online. Moghimi et al. (2024) proposed LIRRN, a fast, coregistration-free RRN method based on brightness segmentation and non-spatial PIFs, with MATLAB scripts available [online](https://github.com/ArminMoghimi/LIRRN/tree/main), however, it is not a comprehensive library. Hessel et al. (2020) introduced a multi-sensor method using quality-based reference selection, RANSAC-based affine corrections, and tone mapping; the code is available [online](https://github.com/chlsl/rrn-multisensor-multidate?tab=readme-ov-file) however, it is no longer maintained. These software do provide a solution for some use cases but fall short of delivering an extensible, easy-to-use tool for RRN across sensors, formats, and workflows.
  
# Algorithms  
The global regression feature performs radiometric normalization across overlapping satellite images by aligning their brightness and contrast globally. It first detects overlapping image pairs and computes per-band statistics (mean, standard deviation) within those regions. Using these statistics, a least-squares regression system is constructed to solve for per-image, per-band scale and offset parameters that minimize radiometric differences in overlaps. This approach aims to minimize brightness and contrast differences across images while preserving global consistency and aligning the spectral profiles of images to a central tendency. The method supports nodata-aware processing for images of irregular shapes and internal gaps, vector Pseudo Invariant Feature masking, custom-weighted mean and standard deviation constraints, efficient windowing and parallelization for large datasets and cloud processing, and progress saving and resumption enabled by stored statistics.  
  
The local block adjustment feature applies block-wise radiometric correction to individual satellite images based on local differences from a reference mosaic. The method divides the combined extent of all input images into spatial blocks and calculates local mean statistics for each block. Each image is then locally adjusted using interpolated adaptive gamma normalization to align with the global reference mosaic. This allows radiometric consistency across spatially heterogeneous scenes on a block scale. The method supports nodata-aware processing for images of irregular shapes and internal gaps, vector Pseudo Invariant Feature masking, multiple strategies for determining block layout, efficient windowing and parallelization for large datasets and cloud processing, and progress saving and resumption enabled by stored block maps.  
  
Various helper functions support the creation of cloud masks, non-vegetation PIFs, generating seamlines, merging images, and basic figures. Cloud masking utilities enable the generation of binary masks using [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask) by Wright et al (2025), followed by post-processing and vectorization. Vegetation masking utilities use NDVI-based thresholds, followed by post-processing and vectorization. The created masks can be used to mask input images or withhold pixels from analysis. Seamline generation utilities use Voronoi-based centerlines, following the methodology of Yuan et al. (2023). Statistical utilities can generate basic figures comparing image spectral profiles before and after matching to evaluate radiometric changes. Raster merging utilities combine the final images into a seamless mosaic.  
# Figures  
![Spectral Comparision](https://raw.githubusercontent.com/spectralmatch/spectralmatch/main/images/matching_histogram.png)  
_Figure 1. Shows the average spectral profile of two WorldView 3 images before and after global to local matching._  
# References  

Canty, M. J., & Nielsen, A. A. (2008). Automatic radiometric normalization of multitemporal satellite imagery with the iteratively re-weighted MAD transformation. _Remote Sensing of Environment_, _112_(3), 1025–1036. [https://doi.org/10.1016/j.rse.2007.07.013](https://doi.org/10.1016/j.rse.2007.07.013)

Chavez, P. S. (1996). _Image-Based Atmospheric Corrections—Revisited and Improved_.

Gan, W., Albanwan, H., & Qin, R. (2021). Radiometric Normalization of Multitemporal Landsat and Sentinel-2 Images Using a Reference MODIS Product Through Spatiotemporal Filtering. _IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing_, _14_, 4000–4013. [https://doi.org/10.1109/JSTARS.2021.3069855](https://doi.org/10.1109/JSTARS.2021.3069855)

Hessel, C., Grompone von Gioi, R., Morel, J. M., Facciolo, G., Arias, P., & de Franchis, C. (2020). RELATIVE RADIOMETRIC NORMALIZATION USING SEVERAL AUTOMATICALLY CHOSEN REFERENCE IMAGES FOR MULTI-SENSOR, MULTI-TEMPORAL SERIES. _ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences_, _V-2–2020_, 845–852. XXIV ISPRS Congress, Commission II (Volume V-2-2020) - 2020 edition. [https://doi.org/10.5194/isprs-annals-V-2-2020-845-2020](https://doi.org/10.5194/isprs-annals-V-2-2020-845-2020)

Moghimi, A., Sadeghi, V., Mohsenifar, A., Celik, T., & Mohammadzadeh, A. (2024). LIRRN: Location-Independent Relative Radiometric Normalization of Bitemporal Remote-Sensing Images. _Sensors_, _24_(7), Article 7. [https://doi.org/10.3390/s24072272](https://doi.org/10.3390/s24072272)

Vorovencii, I., & D.M., M. (2014). Relative radiometric normalization methods: Overview and an application to Landsat images. _Journal of Geodesy and Cadastre, RevCAD_, _17_, 193–200.

Wright, N., Duncan, J. M. A., Callow, J. N., Thompson, S. E., & George, R. J. (2025). Training sensor-agnostic deep learning models for remote sensing: Achieving state-of-the-art cloud and cloud shadow identification with OmniCloudMask. Remote Sensing of Environment, 322, 114694. https://doi.org/10.1016/j.rse.2025.114694

Yu, L., Zhang, Y., Sun, M., Zhou, X., & Liu, C. (2017). An auto-adapting global-to-local color balancing method for optical imagery mosaic. ISPRS Journal of Photogrammetry and Remote Sensing, 132, 1–19. https://doi.org/10.1016/j.isprsjprs.2017.08.002  
  
Yuan, X., Cai, Y., & Yuan, W. (2023). Voronoi Centerline-Based Seamline Network Generation Method. Remote Sensing, 15(4), Article 4. https://doi.org/10.3390/rs15040917
