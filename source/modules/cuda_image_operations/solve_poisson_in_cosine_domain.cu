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



#include "cuda_helper.cuh"

template<typename Pixel>
__global__  void solve_poisson_in_cosine_domain_kernel(Pixel* image,
                              uint width, uint height, uint depth,
                              Pixel* result)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    const Pixel z = floorf(index / (width*height));
    int index_rest = index - z * (width*height);
    const Pixel y = floorf(index_rest / width);
    index_rest = index_rest - y * width;
    const Pixel x = index_rest;

    if(x == 0 && y == 0 && z == 0)
    {
        result[index] = 0;
        return;
    }

    result[index] = image[index]
            /
            (6.0
             - 2.0*cospi(x / width)
             - 2.0*cospi(y / height)
             - 2.0*cospi(z / depth) );
}

template<typename Pixel>
__global__  void solve_poisson_in_cosine_domain_kernel_2D(Pixel* image,
                              uint width, uint height, uint depth,
                              Pixel* result)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    const Pixel z = floorf(index / (width*height));
    int index_rest = index - z * (width*height);
    const Pixel y = floorf(index_rest / width);
    index_rest = index_rest - y * width;
    const Pixel x = index_rest;

    if(x == 0 && y == 0)
    {
        result[index] = 0;
        return;
    }

    result[index] = image[index]
            /
            (4.0
             - 2.0*cospi(x / width)
             - 2.0*cospi(y / height) );
}

template<typename Pixel>
Pixel* solve_poisson_in_cosine_domain_kernel_launch(Pixel* image_host,
                              uint width, uint height, uint depth)
{
    int cuda_device_count;
    cudaCheckError( cudaGetDeviceCount(&cuda_device_count) );

//    printf("found %d cuda devices.\n", cuda_device_count);

    uint voxel_count = width*height*depth;
    dim3 block_dimension(CUDA_BLOCK_DIMENSON);
    dim3 grid_dimension((voxel_count + block_dimension.x - 1) / block_dimension.x);

    Pixel* image, *result;

    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMalloc(&image, size) )
    cudaCheckError( cudaMemcpy(image, image_host, size, cudaMemcpyHostToDevice) )

    cudaCheckError( cudaMalloc(&result, size) )
    cudaCheckError( cudaDeviceSynchronize() );

    if(depth == 1)
    {
        solve_poisson_in_cosine_domain_kernel_2D<<<grid_dimension, block_dimension>>>(
          image, width, height, depth,
          result);
    }
    else
    {
        solve_poisson_in_cosine_domain_kernel<<<grid_dimension, block_dimension>>>(
          image, width, height, depth,
          result);
    }
    cudaCheckError( cudaDeviceSynchronize() );

    Pixel* result_host = new Pixel[voxel_count];
    cudaCheckError( cudaMemcpy(result_host, result, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaDeviceSynchronize() );

    cudaFree(image);
    cudaFree(result);
    cudaCheckError( cudaDeviceSynchronize() );

    return result_host;
}

template float* solve_poisson_in_cosine_domain_kernel_launch(float* image_host,
                              uint width, uint height, uint depth);
template double* solve_poisson_in_cosine_domain_kernel_launch(double* image_host,
                              uint width, uint height, uint depth);
