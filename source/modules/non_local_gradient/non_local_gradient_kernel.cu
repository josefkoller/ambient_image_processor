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



#include "cuda.h"

#include <stdio.h>

#include "cuda_helper.cuh"

template<typename Pixel>
__global__ void non_local_gradient_kernel(Pixel* source,
  uint source_width, uint source_height, uint source_depth, Pixel* kernel, uint kernel_size,
  Pixel* destination) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= source_width * source_height * source_depth)
        return;

    const int z = floorf(index / (source_width*source_height));
    int index_rest = index - z * (source_width*source_height);
    const int y = floorf(index_rest / source_width);
    index_rest = index_rest - y * source_width;
    const int x = index_rest;

    const int kernel_center = floorf(kernel_size / 2.0f);

    destination[index] = 0;
    for(int kz = 0; kz < kernel_size; kz++)
    {
        for(int ky = 0; ky < kernel_size; ky++)
        {
            for(int kx = 0; kx < kernel_size; kx++)
            {
                if(kx == kernel_center && ky == kernel_center && kz == kernel_center)
                    continue;

                const int ki = kz * kernel_size*kernel_size + kx + ky * kernel_size;

                const int kxa = x + (kx - kernel_center);
                const int kya = y + (ky - kernel_center);
                const int kza = z + (kz - kernel_center);

                if(kxa < 0 || kxa >= source_width ||
                   kya < 0 || kya >= source_height ||
                    kza < 0 || kza >= source_depth)
                    continue;

                const int kia = kza * source_width*source_height + kxa + kya * source_width;

    //            printf("kernel value: %f \n", kernel[ki]);

                destination[index] += fabsf(source[index] - source[kia]) * kernel[ki];
            }
        }
    }
}


template<typename Pixel>
Pixel* non_local_gradient_kernel_launch(Pixel* source,
    uint source_width, uint source_height, uint source_depth, Pixel* kernel, uint kernel_size)
{
    int cuda_device_count;
    cudaCheckError( cudaGetDeviceCount(&cuda_device_count) );

//    printf("found %d cuda devices.\n", cuda_device_count);

    const dim3 block_dimension(CUDA_BLOCK_DIMENSON);
    const dim3 grid_dimension(
                (source_width*source_height*source_depth + block_dimension.x - 1) / block_dimension.x);

 //   printf("block dimensions: %d \n", block_dimension.x);
 //   printf("grid dimensions: %d \n", grid_dimension.x);

    Pixel* source_cuda, *kernel_cuda, *destination_cuda;

    size_t size = sizeof(Pixel) * source_width*source_height*source_depth;
    cudaCheckError( cudaMalloc(&source_cuda, size) );
    cudaCheckError( cudaMemcpy(source_cuda, source, size, cudaMemcpyHostToDevice) );

    size_t kernel_size_bytes = sizeof(Pixel) * kernel_size*kernel_size*kernel_size;
    cudaCheckError( cudaMalloc(&kernel_cuda, kernel_size_bytes) );
    cudaCheckError( cudaMemcpy(kernel_cuda, kernel, kernel_size_bytes, cudaMemcpyHostToDevice) );

    cudaCheckError( cudaMalloc(&destination_cuda, size) );

    non_local_gradient_kernel<<<grid_dimension, block_dimension>>>(
      source_cuda, source_width, source_height, source_depth,
      kernel_cuda, kernel_size,
      destination_cuda);
    cudaCheckError( cudaDeviceSynchronize() );

    Pixel* destination = new Pixel[source_width * source_height * source_depth];
    cudaCheckError( cudaMemcpy(destination, destination_cuda, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaDeviceSynchronize() );

    cudaFree(source_cuda);
    cudaFree(kernel_cuda);
    cudaFree(destination_cuda);

    return destination;
}

// generate the algorithm explicitly for...
template float* non_local_gradient_kernel_launch<float>(float* source,
    uint source_width, uint source_height, uint source_depth, float* kernel, uint kernel_size);
template double* non_local_gradient_kernel_launch<double>(double* source,
    uint source_width, uint source_height, uint source_depth, double* kernel, uint kernel_size);
