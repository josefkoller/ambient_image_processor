
#include "unary_operation.cu"

template<typename Pixel>
__global__ void log_kernel(
        Pixel* image,
        const uint width, const uint height, const uint depth)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    image[index] = log(image[index]);
}

template<typename Pixel>
Pixel* log_kernel_launch(Pixel* image_host,
                  uint width, uint height, uint depth)
{
    dim3 block_dimension;
    dim3 grid_dimension;
    Pixel* image;

    unary_operation_part1(image_host,
                      width, height, depth,
                      &image,
                      block_dimension, grid_dimension);

    log_kernel<<<grid_dimension, block_dimension>>>(image, width, height, depth);
    cudaCheckError( cudaDeviceSynchronize() );

    return unary_operation_part2(image, width, height, depth);
}

template float* log_kernel_launch(float* image,
                  uint width, uint height, uint depth);
template double* log_kernel_launch(double* image,
                  uint width, uint height, uint depth);
