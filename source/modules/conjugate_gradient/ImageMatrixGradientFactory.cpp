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

#include "ImageMatrixGradientFactory.h"

ImageMatrixGradientFactory::ImageMatrixGradientFactory()
{

}

template<typename Pixel>
void ImageMatrixGradientFactory::forwardDifferenceX(
        Dimension image_width, Dimension image_height, Dimension image_depth, ImageMatrix<Pixel>* matrix)
{
    matrix->setZeros();

    Dimension z_offset = image_width * image_height;

    for(int z = 0; z < image_depth; z++)
    {
        for(int y = 0; y < image_height; y++)
        {
            for(int x = 0; x < image_width - 1; x++)
            {
                int pixel_index = z * z_offset + (x + y * image_width);
                matrix->setPixelTransformation(pixel_index, pixel_index, -1);
                matrix->setPixelTransformation(pixel_index + 1, pixel_index, 1);
            }
        }
    }
}

template<typename Pixel>
void ImageMatrixGradientFactory::forwardDifferenceY(
        Dimension image_width, Dimension image_height, Dimension image_depth, ImageMatrix<Pixel>* matrix)
{
    matrix->setZeros();

    Dimension z_offset = image_width * image_height;

    for(int z = 0; z < image_depth; z++)
    {
        for(int x = 0; x < image_width; x++)
        {
            for(int y = 0; y < image_height - 1; y++)
            {
                int pixel_index = z * z_offset + (x + y * image_width);
                matrix->setPixelTransformation(pixel_index, pixel_index, -1);
                matrix->setPixelTransformation(pixel_index + image_height, pixel_index, 1);
            }
        }
    }
}

template<typename Pixel>
void ImageMatrixGradientFactory::forwardDifferenceZ(
        Dimension image_width, Dimension image_height, Dimension image_depth, ImageMatrix<Pixel>* matrix)
{
    matrix->setZeros();

    Dimension z_offset = image_width * image_height;

    for(int y = 0; y < image_height; y++)
    {
        for(int x = 0; x < image_width; x++)
        {
            for(int z = 0; z < image_depth - 1; z++)
            {
                int pixel_index = z * z_offset + (x + y * image_width);
                matrix->setPixelTransformation(pixel_index, pixel_index, -1);
                matrix->setPixelTransformation(pixel_index + z_offset, pixel_index, 1);
            }
        }
    }
}

template<typename Pixel>
ImageMatrix<Pixel>* ImageMatrixGradientFactory::laplace(
        Dimension image_width, Dimension image_height, Dimension image_depth)
{
    Dimension voxel_count = image_width * image_height * image_depth;
    auto matrix1 = new ImageMatrix<Pixel>(voxel_count);
    forwardDifferenceX(image_width, image_height, image_depth, matrix1);

    auto matrix2 = new ImageMatrix<Pixel>(voxel_count);
    matrix1->transposed(matrix2);

    auto matrix3 = new ImageMatrix<Pixel>(voxel_count);
    matrix1->multiply(matrix2, matrix3);

    forwardDifferenceY(image_width, image_height, image_depth, matrix1);
    matrix1->transposed(matrix2);
    matrix1->multiply(matrix2, matrix1);
    matrix1->add(matrix3, matrix3);

    forwardDifferenceZ(image_width, image_height, image_depth, matrix1);
    matrix1->transposed(matrix2);
    matrix1->multiply(matrix2, matrix1);
    matrix1->add(matrix3, matrix3);

    return matrix3;

}

typedef const unsigned int Dimension;

template ImageMatrix<double>* ImageMatrixGradientFactory::laplace(
        Dimension image_width, Dimension image_height, Dimension image_depth);

template ImageMatrix<float>* ImageMatrixGradientFactory::laplace(
        Dimension image_width, Dimension image_height, Dimension image_depth);
