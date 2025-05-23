# Dimensions of Relative Radiometric Normalization (RRN) Methods

RRN methods differ not only in the algorithms used to adjust image values but also in the requirements images must have and other techniques that can be used in conjunction. The following taxonomy summarizes the core dimensions along which RRN techniques vary:

 - **Matching algorithm:** The core transformation applied to align radiometry between images.
 - **Geometric alignment required:** The level of spatial alignment necessary for the method.
 - **PIF/RCS selection strategies:** How pseudo-invariant features/control sets are identified.
 - **Adjustment scope:** How corrections are applied to the images.
 - **Overlap:** Whether the method requires overlapping pixels.
 - **Pixel units:** The radiometric units the method is able to operate on.
 - **Bands:** Whether bands relationships are preserved.
 - **Target reference:** What the target image is normalized to.

## The dimensions

- Matching algorithm:
	- Histogram Matching (HM) (lookup table)
	- Minimum–Maximum Normalization (min-max)
	- Mean–Standard Deviation Normalization (gain-offset)
	- CCA/KCCA-Based Normalization (matrix)
	- Global Linear Regression (gain-offset)
	- Gamma correction (power function)
	- Dodging (low-pass brightness correction)
	- Illumination Equalization (modeled lighting correction)
- Minimum geometric alignment:
	- None (no spatial info)
	- Moderate (A few pixels)
	- Co-registration (pixel-wise)
-  Pseudo-Invariant Feature (PIFs)/Radiometric Control Sets (RCS) selection strategies:
    - None
        - Whole image
        - Overlapping area
    - Manual
        - Manual polygons or pixels
        - Manual threshold
    - Statistical
        - Dark/Bright Set (DB)
        - Band indexes
        - No-change  Scattergrams (NC)
        - Multivariate Alteration Detection (MAD)
        - Iteratively Reweighted MAD (IR-MAD)
        - Multi-Rule-Based Normalization
    - Geometric
        - Feature-Based (Keypoint) RRN
        - Location-Independent RRN (LIRRN)
- Adjustment scope:
	- Global
	- Blocks/interpolated blocks
	- CCA space
	- Blur
	- Surface model
- Overlap:
	- Required
	- Not required
- Pixel units:
	- Any
	- Reflectance
	- Radiance
	- DN
- Bands:
	- Independent
	- Correlated
- Target reference:
	- Single image
	- Virtual central tendency
	- Learned distribution
