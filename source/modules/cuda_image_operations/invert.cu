
#include "unary_operation.cu"

template<typename Pixel>
__global__ void invert_kernel(
        Pixel* image,
        const uint width, const uint height, const uint depth)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    image[index] = 1.0 - image[index];
}

template<typename Pixel>
Pixel* invert_kernel_launch(Pixel* image_host,
                  uint width, uint height, uint depth)
{
    dim3 block_dimension;
    dim3 grid_dimension;
    Pixel* image;

    unary_operation_part1(image_host,
                      width, height, depth,
                      &image,
                      block_dimension, grid_dimension);

    invert_kernel<<<grid_dimension, block_dimension>>>(image, width, height, depth);

    return unary_operation_part2(image, width, height, depth);
}

template float* invert_kernel_launch(float* image,
                  uint width, uint height, uint depth);
template double* invert_kernel_launch(double* image,
                  uint width, uint height, uint depth);
