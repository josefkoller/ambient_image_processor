
#include "cuda_helper.cuh"

template<typename Pixel>
__global__  void convolution3x3_kernel(Pixel* image,
                              uint width, uint height,
                              Pixel* kernel, Pixel* result)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height)
        return;

    const int y = floorf(index / width);
    const int x = index - y * width;

    result[index] = image[index] * kernel[4];

    bool is_not_left = x > 0;
    bool is_not_top = y > 0;
    bool is_not_bottom = y < height - 1;
    bool is_not_right = x < width - 1;

    if(is_not_left)
    {
        result[index] += image[index-1] * kernel[3];
        if(is_not_top)
            result[index] += image[index-1-width] * kernel[0];
        if(is_not_bottom)
            result[index] += image[index-1+width] * kernel[6];
    }
    if(is_not_top)
        result[index] += image[index-width] * kernel[1];
    if(is_not_bottom)
        result[index] += image[index+width] * kernel[7];

    if(is_not_right)
    {
        result[index] += image[index+1] * kernel[5];
        if(is_not_top)
            result[index] += image[index+1-width] * kernel[2];
        if(is_not_bottom)
            result[index] += image[index+1+width] * kernel[8];
    }
}

template<typename Pixel>
__global__  void convolution3x3_with_dynamic_center_kernel(Pixel* image,
                              uint width, uint height,
                              Pixel* kernel, Pixel* result)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height)
        return;

    const int y = floorf(index / width);
    const int x = index - y * width;

    result[index] = 0;

    bool is_not_left = x > 0;
    bool is_not_top = y > 0;
    bool is_not_bottom = y < height - 1;
    bool is_not_right = x < width - 1;

    Pixel center_sum = 0;

    if(is_not_left)
    {
        result[index] += image[index-1] * kernel[3]; center_sum+= kernel[3];
        if(is_not_top) {
            result[index] += image[index-1-width] * kernel[0]; center_sum+= kernel[0];
        }
        if(is_not_bottom) {
            result[index] += image[index-1+width] * kernel[6]; center_sum+= kernel[6];
        }
    }
    if(is_not_top) {
        result[index] += image[index-width] * kernel[1]; center_sum+= kernel[1];
    }
    if(is_not_bottom) {
        result[index] += image[index+width] * kernel[7]; center_sum+= kernel[7];
    }

    if(is_not_right)
    {
        result[index] += image[index+1] * kernel[5]; center_sum+= kernel[5];
        if(is_not_top) {
            result[index] += image[index+1-width] * kernel[2]; center_sum+= kernel[2];
        }
        if(is_not_bottom) {
            result[index] += image[index+1+width] * kernel[8]; center_sum+= kernel[8];
        }
    }

    result[index] -= image[index] * center_sum;
}


template<typename Pixel>
Pixel* convolution3x3_kernel_launch(Pixel* image_host,
                              uint width, uint height,
                                    Pixel* kernel_host, bool calculate_center_as_sum_of_others)
{
    int cuda_device_count;
    cudaCheckError( cudaGetDeviceCount(&cuda_device_count) );

//    printf("found %d cuda devices.\n", cuda_device_count);

    uint voxel_count = width*height;
    dim3 block_dimension(CUDA_BLOCK_DIMENSON);
    dim3 grid_dimension((voxel_count + block_dimension.x - 1) / block_dimension.x);

    Pixel* image, *kernel, *result;

    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMalloc(&image, size) )
    cudaCheckError( cudaMemcpy(image, image_host, size, cudaMemcpyHostToDevice) )

    size_t kernel_size = sizeof(Pixel) * 9;
    cudaCheckError( cudaMalloc(&kernel, size) )
    cudaCheckError( cudaMemcpy(kernel, kernel_host, kernel_size, cudaMemcpyHostToDevice) )

    cudaCheckError( cudaMalloc(&result, size) )

    cudaCheckError( cudaDeviceSynchronize() );

    if(!calculate_center_as_sum_of_others)
        convolution3x3_kernel<<<grid_dimension, block_dimension>>>(
          image, width, height,
          kernel, result);
    else
        convolution3x3_with_dynamic_center_kernel<<<grid_dimension, block_dimension>>>(
          image, width, height,
          kernel, result);
    cudaCheckError( cudaDeviceSynchronize() );

    Pixel* result_host = new Pixel[voxel_count];
    cudaCheckError( cudaMemcpy(result_host, result, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaDeviceSynchronize() );

    cudaFree(image);
    cudaFree(kernel);
    cudaFree(result);

    return result_host;
}

template float* convolution3x3_kernel_launch(float* image_host,
                              uint width, uint height,
                              float* kernel_host, bool calculate_center_as_sum_of_others);
template double* convolution3x3_kernel_launch(double* image_host,
                              uint width, uint height,
                              double* kernel_host, bool calculate_center_as_sum_of_others);
