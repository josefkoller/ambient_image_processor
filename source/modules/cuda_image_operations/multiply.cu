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

#include "binary_operation.cu"

template<typename Pixel>
__global__ void multiply_kernel(
        Pixel* image1, Pixel* image2,
        const uint width, const uint height, const uint depth)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    image1[index] = image1[index] * image2[index];
}

template<typename Pixel>
Pixel* multiply_kernel_launch(Pixel* image1_host, Pixel* image2_host,
                  uint width, uint height, uint depth)
{
    dim3 block_dimension;
    dim3 grid_dimension;
    Pixel* image1, *image2;

    binary_operation_part1(image1_host, image2_host,
                      width, height, depth,
                      &image1, &image2,
                      block_dimension, grid_dimension);

    multiply_kernel<<<grid_dimension, block_dimension>>>(
         image1, image2, width, height, depth);
    cudaCheckError( cudaDeviceSynchronize() );

    return binary_operation_part2(image1, image2,
                                  width, height, depth);
}

template float* multiply_kernel_launch(float* image1, float* image2,
                  uint width, uint height, uint depth);
template double* multiply_kernel_launch(double* image1, double* image2,
                  uint width, uint height, uint depth);
