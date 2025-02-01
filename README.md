# Image Processing Utilities

This repository contains a collection of Python utilities for image processing. The utilities are designed to perform various operations on images, such as grayscale conversion, halftoning, histogram equalization, edge detection, filtering, and more. The code is built using popular libraries like `Pillow`, `NumPy`, `OpenCV`, and `Matplotlib`.

## Features

- **Metadata Extraction**: Extract metadata from images, including dimensions, color mode, and EXIF data.
- **Histogram Calculation**: Generate histograms for grayscale and RGB images.
- **Grayscale Conversion**: Convert images to grayscale using different methods (luminosity, lightness).
- **Halftoning**: Apply halftoning techniques to images using thresholding or error diffusion.
- **Histogram Equalization**: Enhance image contrast using histogram equalization.
- **Edge Detection**: Detect edges in images using various operators (Roberts, Sobel, Prewitt, Kirsch, Robinson, Laplacian, etc.).
- **Filtering**: Apply low-pass, high-pass, and median filters to images.
- **Image Operations**: Perform basic image operations like rotation, flipping, resizing, and inversion.
- **Multi-Image Operations**: Combine multiple images using operations like addition, subtraction, and cut-paste.
- **Histogram Segmentation**: Segment images based on histogram analysis.

## Installation

To use these utilities, you need to have Python installed on your system. You can install the required dependencies using `pip`:

```bash
pip install pillow numpy opencv-python matplotlib
