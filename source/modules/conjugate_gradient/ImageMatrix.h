/*
    Ambient Image Processor - A tool to perform several imaging tasks
    
    Copyright (C) 2016 Josef Koller

    https://github.com/josefkoller/ambient_image_processor    
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef IMAGEMATRIX_H
#define IMAGEMATRIX_H

#include <cuda_runtime.h>

template<typename Pixel>
class ImageMatrix
{
private:
    typedef const unsigned int Dimension;

    Dimension element_count;

    dim3 block_dimension;
    dim3 grid_dimension;

public:
    Pixel* elements;
    Dimension voxel_count;

    ImageMatrix(Dimension voxel_count);
    ~ImageMatrix();

    void setZeros();

    void setPixelTransformation(Dimension source_pixel_index,
                                Dimension target_pixel_index,
                                Pixel factor);

    void transposed(ImageMatrix<Pixel>* transposed_matrix);
    void add(ImageMatrix<Pixel>* matrix2, ImageMatrix<Pixel>* result);
    void multiply(ImageMatrix<Pixel>* matrix2, ImageMatrix<Pixel>* result);
};

#endif // IMAGEMATRIX_H
