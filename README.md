# spectralmatch
This plugin performs global and local spectral matching of two or more overlapping images to achieve seamless mosaicking. It uses a least squares regression model in the global optimization step to balance colors across all images, ensuring minimal overall color differences by solving for scale and offset parameters in a single step. After global correction, local optimization refines color balancing in overlapping areas. This step applies a Gamma transform, which adjusts color block-by-block to account for variations that the global method might not fully resolve. By working at a finer scale, this step enhances the seamless blending of mosaicked images while maintaining consistency.

Key Features
- Does not assume a reference image, avoiding bias or distortions.
- Applies both global and local optimization to balance colors effectively.
- Uses least squares optimization to model color balancing across all images.
- Adjusts color through scale and offset, ensuring consistency.
- Handles large-scale datasets through automation and parallelization.
- Minimizes excessive color normalization, preserving real spectral information.
- Sensor-agnostic, making it applicable to different optical remote sensing datasets.

Assumptions
- Overlapping areas of images should have the same spectral profile.
- Color differences can be modeled using least squares fitting.
- Scale and offset parameters effectively adjust image colors.
- The best color correction minimizes differences across images.
- Images are geometrically aligned, with known relative positions.
- Global correction assumes uniform color differences.
- Local correction assumes variations occur due to global adjustments.


Requirements
	•	Python >= 3.10
	•	proj >= 9.3
	•	gdal >= 3.6
	•	All Python packages listed in requirements.txt

Installation
	1.	Clone the repo
	2.	Install system requirements
	3.	Install Python requirements listed in requirements.txt

For Dev

Install dev requirements and run:

pre-commit install
