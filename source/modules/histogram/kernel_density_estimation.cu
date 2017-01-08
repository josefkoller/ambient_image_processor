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

#define PI (3.14159265359)

typedef unsigned int uint;

enum KernelType {
    Uniform = 0,
    Gaussian,
    Cosine,
    Epanechnik
};

template<typename Pixel>
__global__ void kernel_density_estimation_kernel_uniform(
  Pixel* image, Pixel* mask, uint voxel_count,
  Pixel window_from, Pixel window_to,
  Pixel* spectrum, uint spectrum_bandwidth,
  Pixel kernel_bandwidth)
{
    const int spectrum_index = blockDim.x * blockIdx.x + threadIdx.x;

    if(spectrum_index >= spectrum_bandwidth)
        return;

    Pixel spectrum_value = window_from +
            (window_to - window_from) / spectrum_bandwidth * spectrum_index;

    spectrum[spectrum_index] = 0;
    for(uint pixel_index = 0; pixel_index < voxel_count; pixel_index++)
    {
        if(mask != nullptr && mask[pixel_index] == 0)
            continue;

        Pixel pixel_value = image[pixel_index];
        Pixel u = (spectrum_value - pixel_value) / kernel_bandwidth;
        if(fabs(u) >= 1)
            continue;

        spectrum[spectrum_index]++; // uniform kernel
    }
}

template<typename Pixel>
__global__ void kernel_density_estimation_kernel_gaussian(
  Pixel* image, Pixel* mask, uint voxel_count,
  Pixel window_from, Pixel window_to,
  Pixel* spectrum, uint spectrum_bandwidth,
  Pixel kernel_bandwidth)
{
    const int spectrum_index = blockDim.x * blockIdx.x + threadIdx.x;

    if(spectrum_index >= spectrum_bandwidth)
        return;

    Pixel spectrum_value = window_from +
            (window_to - window_from) / spectrum_bandwidth * spectrum_index;

    spectrum[spectrum_index] = 0;
    for(uint pixel_index = 0; pixel_index < voxel_count; pixel_index++)
    {
        if(mask != nullptr && mask[pixel_index] == 0)
            continue;

        Pixel pixel_value = image[pixel_index];
        Pixel u = (spectrum_value - pixel_value) / kernel_bandwidth;
        if(fabs(u) >= 1)
            continue;

        spectrum[spectrum_index]+= exp(-u*u/2) / sqrtf(2*PI);
    }
}

template<typename Pixel>
__global__ void kernel_density_estimation_kernel_cosine(
  Pixel* image, Pixel* mask, uint voxel_count,
  Pixel window_from, Pixel window_to,
  Pixel* spectrum, uint spectrum_bandwidth,
  Pixel kernel_bandwidth)
{
    const int spectrum_index = blockDim.x * blockIdx.x + threadIdx.x;

    if(spectrum_index >= spectrum_bandwidth)
        return;

    Pixel spectrum_value = window_from +
            (window_to - window_from) / spectrum_bandwidth * spectrum_index;

    spectrum[spectrum_index] = 0;
    for(uint pixel_index = 0; pixel_index < voxel_count; pixel_index++)
    {
        if(mask != nullptr && mask[pixel_index] == 0)
            continue;

        Pixel pixel_value = image[pixel_index];
        Pixel u = (spectrum_value - pixel_value) / kernel_bandwidth;
        if(fabs(u) >= 1)
            continue;

        spectrum[spectrum_index]+= PI/4 * cospi(u/2);
    }
}

template<typename Pixel>
__global__ void kernel_density_estimation_kernel_epanechnik(
  Pixel* image, Pixel* mask, uint voxel_count,
  Pixel window_from, Pixel window_to,
  Pixel* spectrum, uint spectrum_bandwidth,
  Pixel kernel_bandwidth)
{
    const int spectrum_index = blockDim.x * blockIdx.x + threadIdx.x;

    if(spectrum_index >= spectrum_bandwidth)
        return;

    Pixel spectrum_value = window_from +
            (window_to - window_from) / spectrum_bandwidth * spectrum_index;

    spectrum[spectrum_index] = 0;
    for(uint pixel_index = 0; pixel_index < voxel_count; pixel_index++)
    {
        if(mask != nullptr && mask[pixel_index] == 0)
            continue;

        Pixel pixel_value = image[pixel_index];
        Pixel u = (spectrum_value - pixel_value) / kernel_bandwidth;
        if(fabs(u) >= 1)
            continue;

        spectrum[spectrum_index]+= 3.0/4.0 * (1-u*u);
    }
}

template<typename Pixel>
Pixel* kernel_density_estimation_kernel_launch(Pixel* image_host, Pixel* mask_host,
                                              uint voxel_count,
                                              uint spectrum_bandwidth,
                                              Pixel kernel_bandwidth,
                                              uint kernel_type,
                                              Pixel window_from,
                                              Pixel window_to)
{
    int cuda_device_count;
    cudaCheckError( cudaGetDeviceCount(&cuda_device_count) );
 //   printf("found %d cuda devices.\n", cuda_device_count);

    const dim3 block_dimension(CUDA_BLOCK_DIMENSON);
    const dim3 grid_dimension(
                (spectrum_bandwidth + block_dimension.x - 1) / block_dimension.x);
 //   printf("block dimensions: %d \n", block_dimension.x);
 //   printf("grid dimensions: %d \n", grid_dimension.x);

    Pixel* image, *mask;
    size_t image_size = sizeof(Pixel) * voxel_count;

    cudaCheckError( cudaMalloc(&image, image_size) );
    cudaCheckError( cudaMemcpy(image, image_host, image_size, cudaMemcpyHostToDevice) );

    if(mask_host == nullptr)
        mask = nullptr;
    else {
        cudaCheckError( cudaMalloc(&mask, image_size) );
        cudaCheckError( cudaMemcpy(mask, mask_host, image_size, cudaMemcpyHostToDevice) );
    }

    Pixel* spectrum;
    size_t spectrum_size = sizeof(Pixel) * spectrum_bandwidth;
    cudaCheckError( cudaMalloc(&spectrum, spectrum_size) );

    if(kernel_type == Uniform)
        kernel_density_estimation_kernel_uniform<<<grid_dimension, block_dimension>>>(
          image, mask, voxel_count,
          window_from, window_to,
          spectrum, spectrum_bandwidth,
          kernel_bandwidth);
    else if(kernel_type == Gaussian)
        kernel_density_estimation_kernel_gaussian<<<grid_dimension, block_dimension>>>(
          image, mask, voxel_count,
          window_from, window_to,
          spectrum, spectrum_bandwidth,
          kernel_bandwidth);
    else if(kernel_type == Cosine)
        kernel_density_estimation_kernel_cosine<<<grid_dimension, block_dimension>>>(
          image, mask, voxel_count,
          window_from, window_to,
          spectrum, spectrum_bandwidth,
          kernel_bandwidth);
    else if(kernel_type == Epanechnik)
        kernel_density_estimation_kernel_epanechnik<<<grid_dimension, block_dimension>>>(
          image, mask, voxel_count,
          window_from, window_to,
          spectrum, spectrum_bandwidth,
          kernel_bandwidth);
    else
    {
        printf("invalid kernel type: %d \n", kernel_type);
        return nullptr;
    }

    cudaCheckError( cudaDeviceSynchronize() );

    Pixel* spectrum_host = new Pixel[spectrum_bandwidth];
    cudaCheckError( cudaMemcpy(spectrum_host, spectrum, spectrum_size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaDeviceSynchronize() );

    cudaFree(mask);
    cudaFree(image);
    cudaFree(spectrum);

    return spectrum_host;
}

template double* kernel_density_estimation_kernel_launch(double* image_pixels, double* mask_pixels,
uint voxel_count,
uint spectrum_bandwidth,
double kernel_bandwidth,
uint kernel_type,
double window_from,
double window_to);

template float* kernel_density_estimation_kernel_launch(float* image_pixels, float* mask_pixels,
uint voxel_count,
uint spectrum_bandwidth,
float kernel_bandwidth,
uint kernel_type,
float window_from,
float window_to);
