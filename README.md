 
A tool to perform several *imaging* tasks.

# Dependencies
- Qt 5.5.0, installed from source, [qt.io](https://www.qt.io/)
- Cuda 7.0, package cuda-7-0
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
