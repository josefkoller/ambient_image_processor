

#include "unary_operation.cu"

template<typename Pixel>
__global__ void clamp_negative_values_kernel(
        Pixel* image,
        const uint width, const uint height, const uint depth, Pixel value)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    if(image[index] < 0)
        image[index] = value;
}

template<typename Pixel>
Pixel* clamp_negative_values_kernel_launch(Pixel* image_host,
                  uint width, uint height, uint depth, Pixel value)
{
    dim3 block_dimension;
    dim3 grid_dimension;
    Pixel* image;

    unary_operation_part1(image_host,
                      width, height, depth,
                      &image,
                      block_dimension, grid_dimension);

    clamp_negative_values_kernel<<<grid_dimension, block_dimension>>>(image, width, height, depth, value);
    cudaCheckError( cudaDeviceSynchronize() );

    return unary_operation_part2(image, width, height, depth);
}

template float* clamp_negative_values_kernel_launch(float* image,
                  uint width, uint height, uint depth, float value);
template double* clamp_negative_values_kernel_launch(double* image,
                  uint width, uint height, uint depth, double value);
