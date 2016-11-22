
#include "unary_operation.cu"

template<typename Pixel>
__global__ void rotate_180_in_plane_kernel(
        Pixel* image,
        const uint width, const uint height, const uint depth, Pixel* result)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    const int z = floorf(index / (width*height));
    int index_rest = index - z * (width*height);
    const int y = floorf(index_rest / width);
    index_rest = index_rest - y * width;
    const int x = index_rest;

    const int y2 = height - y - 1;
    const int x2 = width - x - 1;

    const int index2 = x2 + y2 * width + z * width*height;
    result[index2] = image[index];
}

template<typename Pixel>
Pixel* rotate_180_in_plane_kernel_launch(Pixel* image_host,
                  uint width, uint height, uint depth)
{
    dim3 block_dimension;
    dim3 grid_dimension;
    Pixel* image;
    Pixel* result;

    unary_operation_part1(image_host,
                      width, height, depth,
                      &image,
                      block_dimension, grid_dimension);

    uint voxel_count = width*height*depth;
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMalloc(&result, size) )
    cudaCheckError( cudaDeviceSynchronize() );

    rotate_180_in_plane_kernel<<<grid_dimension, block_dimension>>>(image, width, height, depth, result);
    cudaCheckError( cudaDeviceSynchronize() );

    cudaCheckError( cudaFree(image) );

    return unary_operation_part2(result, width, height, depth);
}

template float* rotate_180_in_plane_kernel_launch(float* image,
                  uint width, uint height, uint depth);
template double* rotate_180_in_plane_kernel_launch(double* image,
                  uint width, uint height, uint depth);
