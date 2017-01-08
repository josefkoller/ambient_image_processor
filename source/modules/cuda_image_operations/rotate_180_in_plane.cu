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


#include "unary_operation.cu"

template<typename Pixel>
__global__ void rotate_180_in_plane_kernel(
        Pixel* image,
        const uint width, const uint height, const uint depth, Pixel* result)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    const int z = floorf(index / (width*height));
    int index_rest = index - z * (width*height);
    const int y = floorf(index_rest / width);
    index_rest = index_rest - y * width;
    const int x = index_rest;

    const int y2 = height - y - 1;
    const int x2 = width - x - 1;

    const int index2 = x2 + y2 * width + z * width*height;
    result[index2] = image[index];
}

template<typename Pixel>
Pixel* rotate_180_in_plane_kernel_launch(Pixel* image_host,
                  uint width, uint height, uint depth)
{
    dim3 block_dimension;
    dim3 grid_dimension;
    Pixel* image;
    Pixel* result;

    unary_operation_part1(image_host,
                      width, height, depth,
                      &image,
                      block_dimension, grid_dimension);

    uint voxel_count = width*height*depth;
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMalloc(&result, size) )
    cudaCheckError( cudaDeviceSynchronize() );

    rotate_180_in_plane_kernel<<<grid_dimension, block_dimension>>>(image, width, height, depth, result);
    cudaCheckError( cudaDeviceSynchronize() );

    cudaCheckError( cudaFree(image) );

    return unary_operation_part2(result, width, height, depth);
}

template float* rotate_180_in_plane_kernel_launch(float* image,
                  uint width, uint height, uint depth);
template double* rotate_180_in_plane_kernel_launch(double* image,
                  uint width, uint height, uint depth);
