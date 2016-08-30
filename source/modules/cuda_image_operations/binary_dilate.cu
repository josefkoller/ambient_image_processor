
#include "unary_operation.cu"

template<typename Pixel>
__global__ void binary_dilate_kernel(
        Pixel* image,
        const uint width, const uint height, const uint depth,
        Pixel* result)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    const int z = floorf(index / (width*height));
    int index_rest = index - z * (width*height);
    const int y = floorf(index_rest / width);
    const int x = index_rest - y * width;

    const Pixel threshold = 0.5;

    // dilate 4-neighbourhood

    result[index] =
       image[index] > threshold ||
       (x > 0 && image[index-1] > threshold) ||
       (x < width-1 && image[index+1] > threshold) ||
       (y > 0 && image[index-width] > threshold) ||
       (y < height-1 && image[index+width] > threshold);
}

template<typename Pixel>
Pixel* binary_dilate_kernel_launch(Pixel* image_host,
                  uint width, uint height, uint depth)
{
    dim3 block_dimension;
    dim3 grid_dimension;
    Pixel* image;

    unary_operation_part1(image_host,
                      width, height, depth,
                      &image,
                      block_dimension, grid_dimension);

    uint voxel_count = width*height*depth;
    Pixel* result;
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMalloc(&result, size) )
    cudaCheckError( cudaDeviceSynchronize() );

    binary_dilate_kernel<<<grid_dimension, block_dimension>>>(image, width, height, depth, result);
    cudaCheckError( cudaDeviceSynchronize() );

    cudaFree(image);
    return unary_operation_part2(result, width, height, depth);
}

template float* binary_dilate_kernel_launch(float* image,
                  uint width, uint height, uint depth);
template double* binary_dilate_kernel_launch(double* image,
                  uint width, uint height, uint depth);
