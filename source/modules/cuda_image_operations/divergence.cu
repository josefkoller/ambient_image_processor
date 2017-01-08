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
#include "add.cu"

template<typename Pixel>
void launch_divergence(
        Pixel* dx, Pixel* dy, Pixel* dz,
        Pixel* dxdx, Pixel* dydy, Pixel* dzdz,

        const uint width, const uint height, const uint depth,

        dim3 block_dimension,
        dim3 grid_dimension,
        dim3 grid_dimension_x,
        dim3 grid_dimension_y,
        dim3 grid_dimension_z)
{
    tgv_launch_backward_differences<Pixel>(
            dxdx, dydy, dzdz,
            dx, dy, dz,
            width, height, depth,
            block_dimension,
            grid_dimension_x,
            grid_dimension_y,
            grid_dimension_z);

    add_kernel<<<grid_dimension, block_dimension>>>(
         dxdx, dydy, width, height, depth);
    cudaCheckError( cudaDeviceSynchronize() );
    if(depth > 1)
    {
        add_kernel<<<grid_dimension, block_dimension>>>(
             dxdx, dzdz, width, height, depth);
        cudaCheckError( cudaDeviceSynchronize() );
    }
}

template<typename Pixel>
Pixel* divergence_kernel_launch(
        Pixel* dx, Pixel* dy, Pixel* dz,
        const uint width, const uint height, const uint depth, bool is_host_data=false)
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

    if(is_host_data)
    {
        uint voxel_count = width*height*depth;
        size_t size = sizeof(Pixel) * voxel_count;

        Pixel *dx2, *dy2, *dz2;
        cudaCheckError( cudaMalloc(&dx2, size) )
        cudaCheckError( cudaMalloc(&dy2, size) )
        if(depth > 1)
          cudaCheckError( cudaMalloc(&dz2, size) )
        cudaCheckError( cudaDeviceSynchronize() );

        cudaCheckError( cudaMemcpy(dx2, dx, size, cudaMemcpyHostToDevice) )
        dx = dx2;
        cudaCheckError( cudaMemcpy(dy2, dy, size, cudaMemcpyHostToDevice) )
        dy = dy2;
        if(depth > 1)
        {
            cudaCheckError( cudaMemcpy(dz2, dz, size, cudaMemcpyHostToDevice) )
            dz = dz2;
        }
        cudaCheckError( cudaDeviceSynchronize() );
    }

    Pixel *dxdx, *dydy, *dzdz;
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMalloc(&dxdx, size) )
    cudaCheckError( cudaMalloc(&dydy, size) )
    if(depth > 1)
      cudaCheckError( cudaMalloc(&dzdz, size) )
    cudaCheckError( cudaDeviceSynchronize() );

    launch_divergence(dx, dy, dz,
                      dxdx, dydy, dzdz,
                      width, height, depth,
                      block_dimension,
                      grid_dimension,
                      grid_dimension_x,
                      grid_dimension_y,
                      grid_dimension_z);
    if(depth > 1)
    {
        cudaFree(dzdz);
    }

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
        if(depth > 1)
            cudaFree(dz);
    }

    return result;
}

template float* divergence_kernel_launch(
float* dx, float* dy, float* dz,
const uint width, const uint height, const uint depth, bool is_host_data);

template double* divergence_kernel_launch(
double* dx, double* dy, double* dz,
const uint width, const uint height, const uint depth, bool is_host_data);
