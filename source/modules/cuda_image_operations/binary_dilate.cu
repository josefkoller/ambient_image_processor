
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
    index_rest = index_rest - y * width;
    const int x = index_rest;

    const Pixel value = image[index];

    if(value < 1e-5)
        return;

    // dilate 4-neighbourhood
    result[index] = value;

    if(x > 0)
        result[index-1] = value;
    if(x < width-1)
        result[index+1] = value;

    if(y > 0)
        result[index-width] = value;
    if(y < height - 1)
        result[index+width] = value;

    if(z > 0)
        result[index-width*height] = value;
    if(z < depth - 1)
        result[index+width*height] = value;
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
    cudaCheckError( cudaMallocManaged(&result, size) )

    binary_dilate_kernel<<<grid_dimension, block_dimension>>>(image, width, height, depth, result);

    cudaFree(image);
    return unary_operation_part2(result, width, height, depth);
}

template float* binary_dilate_kernel_launch(float* image,
                  uint width, uint height, uint depth);
template double* binary_dilate_kernel_launch(double* image,
                  uint width, uint height, uint depth);
