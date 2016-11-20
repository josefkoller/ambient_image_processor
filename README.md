 
A tool to perform several **imaging** tasks.

# Features
- Read, Write all formats supported by ITK
- Read, Write The HSV-value channel of color image formats supported by OpenCV
- Control filters for 3D images via a graphical user interface
- Modules:
 - Image View, Slice Control
 - Statistics
 - Line Profile Plot
 - Histogram based on Kernel Density Estimation
 - Binary Image Operations (Add, Subtract, Divide, Multiply)
 - Unary Image Operations (Invert, Binarize, DCT, iDCT, log, exp)
 - (Hard) Thresholding
 - Morphological Filter (Binary Dilate)
 - 3x3x3 Convolution
 - Rescale Intensities
 - Shrink
 - Extract
 - Resize (Nearest Neighbour, Linear, Sinc, BSpline - Interpolation)
 - Non-Local Gradient
 - Region Growing Segmentation
 - Bilateral Filter
 - TGV Denoising
 - Unsharp Masking
 - Multiscale Retinex
 - DCT Poisson Solver
 - TGV-DCT Image Shading Correction

# Dependencies
- Qt 5.7.0 (Qt5Core, Qt5Widgets, Qt5Charts), installed from source, [qt.io](https://www.qt.io/)
- Cuda 8.0, package cuda-8-0
- ITK 4.8.1, installed from source, [itk.org](https://itk.org/)
- OpenCV 3.1.0, installed from source, [opencv.org](http://opencv.org/)
- FFTW 3, package fftw3-dev
- SQLite 3, package libsqlite3-dev

# Build
```
mkdir build
cd build
cmake ..
make
```
The generated binaries are located in the build/output directory.

# Run
```
build/output/tgv2_deshade_application
```
outputs the list of parameters.
Batch processing can be done by using the 
- bash script bash/tgvk_deshade_downsampled_batch.sh
- ruby scripts in the folder ruby
- octave/matlab script octave/tgv2_deshade.m

the gui in executed by running 
```
build/output/ambient_application
```

# TODO
- easier disabling/enabling of modules
- module template, creator script
- add more imaging filters
- module ideas
 - image zoom for view only
 - load volume of multiple single files using the itk dicom reader, sustain meta info
 - load/scan all files of an directory, use dicom reader, group by date, serie, listview, load-button, do not load all data - only the file-paths for each volume, sustain meta info

