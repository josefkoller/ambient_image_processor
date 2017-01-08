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

#include "tgv_common_2d.cu"

template<typename Pixel>
__global__ void add_kernel_2d(
        Pixel* q_xx, Pixel* q_yx,
        uint width, uint height) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width*height)
        return;

    q_xx[index] = q_xx[index] + q_yx[index];
}

template<typename Pixel>
void launch_divergence_2d(
        Pixel* dx, Pixel* dy,
        Pixel* dxdx, Pixel* dydy,

        const uint width, const uint height,

        dim3 block_dimension,
        dim3 grid_dimension,
        dim3 grid_dimension_x,
        dim3 grid_dimension_y)
{
    tgv_launch_backward_differences_2d<Pixel>(
            dxdx, dydy,
            dx, dy,
            width, height,
            block_dimension,
            grid_dimension_x,
            grid_dimension_y);

    add_kernel_2d<<<grid_dimension, block_dimension>>>(
         dxdx, dydy, width, height);
    cudaCheckError( cudaDeviceSynchronize() );
}

template<typename Pixel>
Pixel* divergence_2d_kernel_launch(
        Pixel* dx, Pixel* dy,
        const uint width, const uint height, bool is_host_data=false)
{
    uint voxel_count;
    dim3 block_dimension;
    dim3 grid_dimension;
    dim3 grid_dimension_x;
    dim3 grid_dimension_y;

    tgv_launch_part1_2d<Pixel>(
                        width, height,
                        voxel_count,
                        block_dimension,
                        grid_dimension,
                        grid_dimension_x,
                        grid_dimension_y);

    if(is_host_data)
    {
        uint voxel_count = width*height;
        size_t size = sizeof(Pixel) * voxel_count;

        Pixel *dx2, *dy2;
        cudaCheckError( cudaMalloc(&dx2, size) )
        cudaCheckError( cudaMalloc(&dy2, size) )
        cudaCheckError( cudaDeviceSynchronize() );

        cudaCheckError( cudaMemcpy(dx2, dx, size, cudaMemcpyHostToDevice) )
        dx = dx2;
        cudaCheckError( cudaMemcpy(dy2, dy, size, cudaMemcpyHostToDevice) )
        dy = dy2;
        cudaCheckError( cudaDeviceSynchronize() );
    }

    Pixel *dxdx, *dydy;
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMalloc(&dxdx, size) )
    cudaCheckError( cudaMalloc(&dydy, size) )
    cudaCheckError( cudaDeviceSynchronize() );

    launch_divergence_2d(dx, dy,
                      dxdx, dydy,
                      width, height,
                      block_dimension,
                      grid_dimension,
                      grid_dimension_x,
                      grid_dimension_y);

    Pixel* result = new Pixel[voxel_count];
    cudaCheckError( cudaMemcpy(result, dxdx, size, cudaMemcpyDeviceToHost) )
    cudaCheckError( cudaDeviceSynchronize() );

    cudaFree(dxdx);
    cudaFree(dydy);
    cudaCheckError( cudaDeviceSynchronize() );

    if(is_host_data)
    {
        cudaFree(dx);
        cudaFree(dy);
    }

    return result;
}

template float* divergence_2d_kernel_launch(
float* dx, float* dy,
const uint width, const uint height, bool is_host_data);

template double* divergence_2d_kernel_launch(
double* dx, double* dy,
const uint width, const uint height, bool is_host_data);
