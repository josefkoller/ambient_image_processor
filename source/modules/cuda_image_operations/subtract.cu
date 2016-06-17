

#include "binary_operation.cu"

template<typename Pixel>
__global__ void subtract_kernel(
        Pixel* image1, Pixel* image2,
        const uint width, const uint height, const uint depth)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    image1[index] = image1[index] - image2[index];
}

template<typename Pixel>
Pixel* subtract_kernel_launch(Pixel* image1_host, Pixel* image2_host,
                  uint width, uint height, uint depth)
{
    dim3 block_dimension;
    dim3 grid_dimension;
    Pixel* image1, *image2;

    binary_operation_part1(image1_host, image2_host,
                      width, height, depth,
                      &image1, &image2,
                      block_dimension, grid_dimension);

    subtract_kernel<<<grid_dimension, block_dimension>>>(
         image1, image2, width, height, depth);
    cudaCheckError( cudaDeviceSynchronize() );

    return binary_operation_part2(image1, image2,
                                  width, height, depth);
}

template float* subtract_kernel_launch(float* image1, float* image2,
                  uint width, uint height, uint depth);
template double* subtract_kernel_launch(double* image1, double* image2,
                  uint width, uint height, uint depth);

