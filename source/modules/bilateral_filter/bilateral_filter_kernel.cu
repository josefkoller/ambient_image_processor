

#include "cuda.h"

#include <stdio.h>

#include "cuda_helper.cuh"

template<typename Pixel>
__global__ void bilateral_filter_kernel(Pixel* source,
  uint source_width, uint source_height, uint source_depth, uint kernel_size,
  Pixel sigma_spatial_distance, Pixel sigma_intensity_distance,
  Pixel* destination) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= source_width*source_height*source_depth)
        return;

    const int z = floorf(index / (source_width*source_height));
    int index_rest = index - z * (source_width*source_height);
    const int y = floorf(index_rest / source_width);
    index_rest = index_rest - y * source_width;
    const int x = index_rest;

    const int kernel_center = floorf(kernel_size / 2.0f);

    destination[index] = 0;
    Pixel kernel_sum = 0;
    for(int kz = 0; kz < kernel_size; kz++)
    {
        for(int ky = 0; ky < kernel_size; ky++)
        {
            for(int kx = 0; kx < kernel_size; kx++)
            {
                if(kx == kernel_center && ky == kernel_center && kz == kernel_center)
                    continue;

                const int dx = kx - kernel_center;
                const int dy = ky - kernel_center;
                const int dz = kz - kernel_center;

                const int kxa = x + dx;
                const int kya = y + dy;
                const int kza = z + dz;

                if(kxa < 0 || kxa >= source_width ||
                   kya < 0 || kya >= source_height ||
                    kza < 0 || kza >= source_depth)
                    continue;

                int kia = kza * source_width*source_height + kxa + kya * source_width;

                const Pixel radius_square = dx*dx + dy*dy + dz*dz;
                const Pixel intensity_distance = source[index] - source[kia];
                const Pixel kernel_value =
                      expf(-radius_square / sigma_spatial_distance
                           -intensity_distance*intensity_distance / sigma_intensity_distance);

                kernel_sum += kernel_value;

                destination[index] += kernel_value * source[kia];

            }
        }
    }

    destination[index] /= kernel_sum;
}


template<typename Pixel>
Pixel* bilateral_filter_kernel_launch(Pixel* source,
    uint source_width, uint source_height, uint source_depth, uint kernel_size,
    Pixel sigma_spatial_distance,
    Pixel sigma_intensity_distance)
{
    int cuda_device_count;
    cudaCheckError( cudaGetDeviceCount(&cuda_device_count) );

    printf("found %d cuda devices.\n", cuda_device_count);

    const uint voxel_count = source_width*source_height*source_depth;

    const dim3 block_dimension(CUDA_BLOCK_DIMENSON);
    const dim3 grid_dimension(
                (voxel_count + block_dimension.x - 1) / block_dimension.x);

    printf("block dimensions: x:%d, y:%d \n", block_dimension.x);
    printf("grid dimensions: x:%d, y:%d \n", grid_dimension.x);

    Pixel* source_cuda, *kernel_cuda, *destination_cuda;

    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMalloc(&source_cuda, size) );
    cudaCheckError( cudaMemcpy(source_cuda, source, size, cudaMemcpyHostToDevice) );

    cudaCheckError( cudaMalloc(&destination_cuda, size) );

    bilateral_filter_kernel<<<grid_dimension, block_dimension>>>(
      source_cuda, source_width, source_height, source_depth,
      kernel_size,
      sigma_spatial_distance, sigma_intensity_distance,
      destination_cuda);
    cudaCheckError( cudaDeviceSynchronize() );

    Pixel* destination = new Pixel[voxel_count];
    cudaCheckError( cudaMemcpy(destination, destination_cuda, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaDeviceSynchronize() );

    cudaFree(source_cuda);
    cudaFree(destination_cuda);

    return destination;
}

// generate the algorithm explicitly for...
template float* bilateral_filter_kernel_launch<float>(float* source,
    uint source_width, uint source_height, uint source_depth, uint kernel_size,
    float sigma_spatial_distance,
    float sigma_intensity_distance);
template double* bilateral_filter_kernel_launch<double>(double* source,
    uint source_width, uint source_height, uint source_depth, uint kernel_size,
    double sigma_spatial_distance,
    double sigma_intensity_distance);
