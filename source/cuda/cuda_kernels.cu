
#include "cuda.h"

#include "RawImage.h"

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

__global__ void clone_image(float* source_image,
                            float* destination_image,
                            uint size_x,
                            uint size_y) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= size_x ||
       y >= size_y)
    {
        return;
    }

    int index = x + y * size_x;
    destination_image[index] = source_image[index];
}


extern "C" void clone_image_launch(RawImage::Pointer source,
                                   RawImage::Pointer destination)
{
    int cuda_device_count;
    cudaCheckError( cudaGetDeviceCount(&cuda_device_count) );

    printf("found %d cuda devices.\n", cuda_device_count);

    const dim3 block_dimension(128,128);
    const dim3 grid_dimension(
                (source->size.x + block_dimension.x - 1) / block_dimension.x,
                (source->size.y + block_dimension.y - 1) / block_dimension.y );

    printf("block dimensions: x:%d, y:%d \n", block_dimension.x, block_dimension.y);
    printf("grid dimensions: x:%d, y:%d \n", grid_dimension.x, grid_dimension.y);

    clone_image<<<grid_dimension, block_dimension>>>(
      source->pixel_pointer,
      destination->pixel_pointer,
      source->size.x,
      source->size.y);

    cudaCheckError( cudaDeviceSynchronize() );
}
