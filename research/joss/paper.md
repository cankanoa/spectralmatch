---
title: 'Spectralmatch: relative radiometric normalization toolkit for raster mosaics and time series'
tags:
  - Relative radiometric normalization
  - Python
  - Time series
  - Mosaic
  - QGIS plugin
  - Cloud masking
  - Pseudo invariant features
  - Remote sensing
  - Histogram matching
  - Seamline
authors:
  - name: Kanoa Lindiwe
    orcid: 0009-0009-5520-1911
    affiliation: 1, 2
  - name: Joseph Emile Honour Percival
    orcid: 0000-0001-5941-4601
    affiliation: 1, 2
  - name: Ryan Perroy
    orcid: 0000-0002-4210-3281
    affiliation: 1, 2
affiliations:
 - name: Dept of Geography & Environmental Science, University of Hawaii at Hilo, United States
   index: 1
   ror: 02mp2av58
 - name: Spatial Data Analysis & Visualization Research Lab, University of Hawaii at Hilo, United States
   index: 2
date: 8 May 2025
bibliography: paper.bib
---
# Summary  
Spectralmatch provides algorithms to perform relative radiometric normalization (RRN) to enhance spectral consistency across raster mosaics and time series. It is built for geoscientific use, with a sensor- and unit-agnostic design, optimized for automation and efficiency on arbitrarily many images and bands, and works well with Very High Resolution Imagery (VHRI) as it does not require pixel co-registration. Its current matching algorithms are inspired by @Yu:2017, which include global regression and local block adjustment that minimize inter-image variability without relying on ancillary data. The impact of these functions on spectral consistency is illustrated in \autoref{fig:1}. The software supports cloud and vegetation masking, pseudo invariant feature (PIF) based exclusion, seamline network generation, raster merging, and plotting statistics. The toolkit is available as an open-source Python library, command line interface, and QGIS plugin.
  
# Statement of Need  
Remote sensing relies on mosaics to broaden spatial coverage by merging images across space, and on time series to extend temporal coverage by linking images over time. However, both are affected by inter-image spectral variability, caused by atmospheric, illumination, and/or surface conditions, acquisition geometry, adjacency effect, and other complications [@Theiler:2019]. These factors introduce inconsistencies, reduce accuracy in image analysis, and complicate the detection of actual environmental changes. To address these issues, researchers have explored two correction methods in the literature: absolute radiometric correction and RRN [@Hu:2011]. The absolute approach compensates for errors using in-situ measurements, which may not exist or be difficult to obtain, particularly for historic images [@Canty:2004]. Conversely, the relative approach applies algorithms to minimize apparent spectral differences between images, matching them for consistent analysis, rather than determining true spectral values or relying on ancillary data.

Researchers have examined various algorithms for performing RRN [@Vorovencii:2014], with model selection and the identification of PIFs recognized as among the most critical and challenging aspects [@Hessel:2020]. Most researched RRN methods are not integrated into software packages which leaves subsequent researchers either spending significant time implementing their own versions of the algorithms or relying on the limited available tools. There are commercial software programs that implement RRN which include [ArcGIS Pro](https://pro.arcgis.com/en/pro-app/latest/tool-reference/data-management/color-balance-mosaic-dataset.htm) (dodging, global fit, histogram, and standard deviation), [ENVI](https://www.nv5geospatialsoftware.com/docs/MosaicSeamless.html) (histogram matching), [ERDAS IMAGINE Mosaic Pro](https://supportsi.hexagon.com/s/article/Create-a-Mosaic-using-ERDAS-IMAGINE-MosaicPro?language=en_US) (illumination equalizing, dodging, color balancing, and histogram matching) and others. In addition, open source solutions include QGIS ([histogram matching](https://github.com/Gustavoohs/HistMatch) and [Iteratively Reweighted Multivariate Alteration Detection (IR-MAD)](https://github.com/SMByC/ArrNorm)), MATLAB scripts (pixel similarity grouping by @Moghimi:2024), Python scripts (multi-sensor normalization by @Hessel:2020) and the R 'landsat' library (histogram matching, pixel-wise linear regression, K-T ratio, and urban materials ratio) by @Goslee:2011. While existing solutions cover many use cases, there is not an open source, scalable library to match non-co-registered VHRI using the mean-standard deviation method which this library specifically addresses. In addition, this library provides an extensible structure to add new RRN methods to meet researchers' varying needs and dataset requirements.
  
# Implemented RRN Methods
The current matching algorithm uses a two-step approach involving global regression and local block adjustment following the methods of @Yu:2017. The global regression algorithm adjusts brightness and contrast across overlapping images to reduce spectral differences. It first detects overlapping image pairs and computes per-band statistics (mean and standard deviation) within those regions. Using these statistics, a least-squares regression system is constructed to solve for per-image, per-band scale and offset parameters that minimize radiometric differences in overlapping areas. This approach aims to minimize brightness and contrast differences across images while preserving global consistency and aligning the spectral profiles of images to a central tendency, specific image, or set of images via custom-weighted mean and standard deviation constraints.
  
The local block adjustment algorithm applies block-wise radiometric correction to individual satellite images based on local differences from a reference mosaic. The method divides the combined extent of all input images into spatial blocks and calculates local mean statistics for each block. Each image is then locally adjusted using interpolated adaptive gamma normalization to align with the global reference mosaic. This allows radiometric consistency across spatially heterogeneous scenes on a block scale. Both global and local algorithms support nodata-aware processing for images of irregular shapes and internal gaps, vector PIF masking, efficient windowing and parallelization for large datasets and cloud processing, and saving and loading of intermediate steps.

Various helper functions support the creation of cloud masks, non-vegetation PIFs, generating seamline networks, merging images, and basic figures. Cloud masking utilities enable the generation of binary masks using [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask) by @Wright:2025, followed by post-processing and vectorization. Vegetation masking utilities use NDVI-based thresholds, followed by post-processing and vectorization. The created masks can be used to mask input images or withhold pixels from analysis. Seamline generation utilities use Voronoi-based centerlines, following the methodology of @Yuan:2023. Statistical utilities can generate basic figures comparing image spectral profiles before and after matching to evaluate radiometric changes. Raster merging utilities combine the final images into a seamless mosaic.

# Figures
![Mean spectral profiles of five cloud masked Landsat images before and after applying Normalized Difference Vegetation Index masking, global regression, and local block adjustment with spectralmatch.\label{fig:1}](https://raw.githubusercontent.com/spectralmatch/spectralmatch/main/images/matching_histogram.png)

# References