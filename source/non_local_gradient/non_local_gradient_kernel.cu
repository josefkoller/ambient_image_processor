

#include "cuda.h"

#include <stdio.h>

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t error_code, const char *file, int line, bool abort=true)
{
   if (error_code != cudaSuccess)
   {
      printf("cuda error: %s %s %d\n", cudaGetErrorString(error_code), file, line);
      if (abort)
      {
          exit(error_code);
      }
   }
}

__global__ void non_local_gradient_kernel(float* source,
  uint source_width, uint source_height, float* kernel, uint kernel_size,
  float* destination) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= source_width * source_height)
        return;

    const int y = floorf(index / source_width);
    const int x = index - y * source_width;
    const int kernel_center = floorf(kernel_size / 2.0f);

    destination[index] = 0;
    for(int ky = 0; ky < kernel_size; ky++)
    {
        for(int kx = 0; kx < kernel_size; kx++)
        {
            if(kx == kernel_center && ky == kernel_center)
                continue;

            const int ki = kx + ky * kernel_size;

            const int kxa = x + (kx - kernel_center);
            const int kya = y + (ky - kernel_center);

            if(kxa < 0 || kxa >= source_width ||
               kya < 0 || kya >= source_height)
                continue;

            const int kia = kxa + kya * source_width;

//            printf("kernel value: %f \n", kernel[ki]);

            destination[index] += fabsf(source[index] - source[kia]) * kernel[ki];
        }
    }
}


extern "C" float* non_local_gradient_kernel_launch(float* source,
    uint source_width, uint source_height, float* kernel, uint kernel_size)
{
    int cuda_device_count;
    cudaCheckError( cudaGetDeviceCount(&cuda_device_count) );

    printf("found %d cuda devices.\n", cuda_device_count);

    const dim3 block_dimension(128);
    const dim3 grid_dimension(
                (source_width*source_height + block_dimension.x - 1) / block_dimension.x);

    printf("block dimensions: x:%d, y:%d \n", block_dimension.x);
    printf("grid dimensions: x:%d, y:%d \n", grid_dimension.x);

    float* source_cuda, *kernel_cuda, *destination_cuda;

    size_t size = sizeof(float) * source_width*source_height;
    cudaCheckError( cudaMalloc(&source_cuda, size) );
    cudaCheckError( cudaMemcpy(source_cuda, source, size, cudaMemcpyHostToDevice) );

    size_t kernel_size_bytes = sizeof(float) * kernel_size*kernel_size;
    cudaCheckError( cudaMalloc(&kernel_cuda, kernel_size_bytes) );
    cudaCheckError( cudaMemcpy(kernel_cuda, kernel, kernel_size_bytes, cudaMemcpyHostToDevice) );

    cudaCheckError( cudaMalloc(&destination_cuda, size) );

    non_local_gradient_kernel<<<grid_dimension, block_dimension>>>(
      source_cuda, source_width, source_height,
      kernel_cuda, kernel_size,
      destination_cuda);
    cudaCheckError( cudaDeviceSynchronize() );

    float* destination = new float[source_width * source_height];
    cudaCheckError( cudaMemcpy(destination, destination_cuda, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaDeviceSynchronize() );

    cudaFree(source_cuda);
    cudaFree(kernel_cuda);
    cudaFree(destination_cuda);

    return destination;
}
