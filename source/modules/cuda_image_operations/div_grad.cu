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
#include "tgv_common.cu"


template<typename Pixel>
void launch_divergence(
        Pixel* dx, Pixel* dy, Pixel* dz,
        Pixel* dxdx, Pixel* dydy, Pixel* dzdz,

        const uint width, const uint height, const uint depth,

        dim3 block_dimension,
        dim3 grid_dimension,
        dim3 grid_dimension_x,
        dim3 grid_dimension_y,
        dim3 grid_dimension_z);

template<typename Pixel>
Pixel* div_grad_kernel_launch(Pixel* image_host,
                  uint width, uint height, uint depth)
{
    dim3 block_dimension;
    dim3 grid_dimension;
    Pixel* image;

    unary_operation_part1(image_host,
                      width, height, depth,
                      &image,
                      block_dimension, grid_dimension);

    Pixel* grad_x, *grad_y, *grad_z;
    Pixel *dgrad_y, *dgrad_z;
    uint voxel_count = width*height*depth;
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMalloc(&grad_x, size) )
    cudaCheckError( cudaMalloc(&grad_y, size) )
    cudaCheckError( cudaMalloc(&grad_z, size) )
    cudaCheckError( cudaMalloc(&dgrad_y, size) )
    cudaCheckError( cudaMalloc(&dgrad_z, size) )

    dim3 grid_dimension_x = dim3((depth*height + block_dimension.x - 1) / block_dimension.x);
    dim3 grid_dimension_y = dim3((depth*width + block_dimension.x - 1) / block_dimension.x);
    dim3 grid_dimension_z = dim3((width*height + block_dimension.x - 1) / block_dimension.x);

    tgv_launch_forward_differences(image, grad_x, grad_y, grad_z,
                                   width, height, depth,
                                   block_dimension,
                                   grid_dimension_x, grid_dimension_y, grid_dimension_z);

    launch_divergence(grad_x, grad_y, grad_z,
                      image, dgrad_y, dgrad_z,
                      width, height, depth,
                      block_dimension,
                      grid_dimension,
                      grid_dimension_x,
                      grid_dimension_y,
                      grid_dimension_z);

    cudaFree(grad_x);
    cudaFree(grad_y);
    cudaFree(grad_z);
    cudaFree(dgrad_y);
    cudaFree(dgrad_z);

    return unary_operation_part2(image, width, height, depth);
}

template float* div_grad_kernel_launch(float* image,
                  uint width, uint height, uint depth);
template double* div_grad_kernel_launch(double* image,
                  uint width, uint height, uint depth);

