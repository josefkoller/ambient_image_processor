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


#include "tgv_common.cu"

template<typename Pixel>
__global__ void total_variation(
        Pixel* p_x, Pixel* p_y, Pixel* p_z, const uint width, const uint height, const uint depth) {

    const uint index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width*height*depth)
        return;

    p_x[index] = abs(p_x[index]) + abs(p_y[index]);

    if(depth > 1)
        p_x[index] += abs(p_z[index]);
}


template<typename Pixel>
double tv_kernel_launch(Pixel* f_host,
                        uint width, uint height, uint depth)
{
    uint voxel_count;
    dim3 block_dimension;
    dim3 grid_dimension;
    dim3 grid_dimension_x;
    dim3 grid_dimension_y;
    dim3 grid_dimension_z;

    tgv_launch_part1<Pixel>(
                        width, height, depth,
                        voxel_count,
                        block_dimension,
                        grid_dimension,
                        grid_dimension_x,
                        grid_dimension_y,
                        grid_dimension_z);

    Pixel *f, *p_x, *p_y, *p_z;
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMalloc(&f, size) )
    cudaCheckError( cudaMemcpy(f, f_host, size, cudaMemcpyHostToDevice) )

    cudaCheckError( cudaMalloc(&p_x, size) )
    cudaCheckError( cudaMalloc(&p_y, size) )
    cudaCheckError( cudaMalloc(&p_z, size) )

    cudaCheckError( cudaDeviceSynchronize() );

    tgv_launch_forward_differences<Pixel>(f,
            p_x, p_y, p_z,
            width, height, depth,
            block_dimension,
            grid_dimension_x,
            grid_dimension_y,
            grid_dimension_z);
    cudaCheckError( cudaDeviceSynchronize() );

    total_variation<<<grid_dimension, block_dimension>>>(
          p_x, p_y, p_z, width, height, depth);
    cudaCheckError( cudaDeviceSynchronize() );

    cudaCheckError( cudaMemcpy(f_host, p_x, size, cudaMemcpyDeviceToHost) )

    cudaFree(f);
    cudaFree(p_x);
    cudaFree(p_y);
    cudaFree(p_z);

    double total_variation = 0;
    for(int i = 0; i < voxel_count; i++)
        total_variation += f_host[i];
    return total_variation;
}

template double tv_kernel_launch(float* image_host,
                              uint width, uint height, uint depth);
template double tv_kernel_launch(double* image_host,
                              uint width, uint height, uint depth);
