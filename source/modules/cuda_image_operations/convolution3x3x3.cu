

#include "cuda_helper.cuh"

template<typename Pixel>
__global__  void convolution3x3x3_kernel(Pixel* image,
                              uint width, uint height, uint depth,
                              Pixel* kernel, Pixel* result)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    const int z = floorf(index / (width*height));
    int index_rest = index - z * (width*height);
    const int y = floorf(index_rest / width);
    index_rest = index_rest - y * width;
    const int x = index_rest;

    bool is_not_left = x > 0;
    bool is_not_top = y > 0;
    bool is_not_bottom = y < height - 1;
    bool is_not_right = x < width - 1;
    bool is_not_front = z > 0;
    bool is_not_back = z < depth -1;

    result[index] = 0;
    // first layer
    if(is_not_front)
    {
        const uint index_offset = -width*height;
        const uint kernel_offset = 0;
        result[index] += image[index+index_offset] * kernel[kernel_offset+4];
        if(is_not_left)
        {
            result[index] += image[index+index_offset-1] * kernel[kernel_offset+3];
            if(is_not_top)
                result[index] += image[index+index_offset-1-width] * kernel[kernel_offset];
            if(is_not_bottom)
                result[index] += image[index+index_offset-1+width] * kernel[kernel_offset+6];
        }
        if(is_not_top)
            result[index] += image[index+index_offset-width] * kernel[kernel_offset+1];
        if(is_not_bottom)
            result[index] += image[index+index_offset+width] * kernel[kernel_offset+7];
        if(is_not_right)
        {
            result[index] += image[index+index_offset+1] * kernel[kernel_offset+5];
            if(is_not_top)
                result[index] += image[index+index_offset+1-width] * kernel[kernel_offset+2];
            if(is_not_bottom)
                result[index] += image[index+index_offset+1+width] * kernel[kernel_offset+8];
        }
    }

    if(is_not_back)
    {
        const uint index_offset = width*height;
        const uint kernel_offset = 18;
        result[index] += image[index+index_offset] * kernel[kernel_offset+4];
        if(is_not_left)
        {
            result[index] += image[index+index_offset-1] * kernel[kernel_offset+3];
            if(is_not_top)
                result[index] += image[index+index_offset-1-width] * kernel[kernel_offset];
            if(is_not_bottom)
                result[index] += image[index+index_offset-1+width] * kernel[kernel_offset+6];
        }
        if(is_not_top)
            result[index] += image[index+index_offset-width] * kernel[kernel_offset+1];
        if(is_not_bottom)
            result[index] += image[index+index_offset+width] * kernel[kernel_offset+7];
        if(is_not_right)
        {
            result[index] += image[index+index_offset+1] * kernel[kernel_offset+5];
            if(is_not_top)
                result[index] += image[index+index_offset+1-width] * kernel[kernel_offset+2];
            if(is_not_bottom)
                result[index] += image[index+index_offset+1+width] * kernel[kernel_offset+8];
        }
    }

    // center layer
    const uint index_offset = 0;
    const uint kernel_offset = 9;

    result[index] += image[index+index_offset] * kernel[kernel_offset+4];
    if(is_not_left)
    {
        result[index] += image[index+index_offset-1] * kernel[kernel_offset+3];
        if(is_not_top)
            result[index] += image[index+index_offset-1-width] * kernel[kernel_offset];
        if(is_not_bottom)
            result[index] += image[index+index_offset-1+width] * kernel[kernel_offset+6];
    }
    if(is_not_top)
        result[index] += image[index+index_offset-width] * kernel[kernel_offset+1];
    if(is_not_bottom)
        result[index] += image[index+index_offset+width] * kernel[kernel_offset+7];
    if(is_not_right)
    {
        result[index] += image[index+index_offset+1] * kernel[kernel_offset+5];
        if(is_not_top)
            result[index] += image[index+index_offset+1-width] * kernel[kernel_offset+2];
        if(is_not_bottom)
            result[index] += image[index+index_offset+1+width] * kernel[kernel_offset+8];
    }

}

// the kernel center element is calculated by summing up the used elements
template<typename Pixel>
__global__  void convolution3x3x3_calculate_center_kernel(Pixel* image,
                              uint width, uint height, uint depth,
                              Pixel* kernel, Pixel* result)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    const int z = floorf(index / (width*height));
    int index_rest = index - z * (width*height);
    const int y = floorf(index_rest / width);
    index_rest = index_rest - y * width;
    const int x = index_rest;

    bool is_not_left = x > 0;
    bool is_not_top = y > 0;
    bool is_not_bottom = y < height - 1;
    bool is_not_right = x < width - 1;
    bool is_not_front = z > 0;
    bool is_not_back = z < depth -1;

    Pixel center = 0;

    result[index] = 0;
    // first layer
    if(is_not_front)
    {
        const uint index_offset = -width*height;
        const uint kernel_offset = 0;
        result[index] += image[index+index_offset] * kernel[kernel_offset+4];
        center +=  kernel[kernel_offset+4];
        if(is_not_left)
        {
            result[index] += image[index+index_offset-1] * kernel[kernel_offset+3];
            center +=  kernel[kernel_offset+3];
            if(is_not_top)
            {
                result[index] += image[index+index_offset-1-width] * kernel[kernel_offset];
                center +=  kernel[kernel_offset];
            }
            if(is_not_bottom)
            {
                result[index] += image[index+index_offset-1+width] * kernel[kernel_offset+6];
                center +=  kernel[kernel_offset+6];
            }
        }
        if(is_not_top)
        {
            result[index] += image[index+index_offset-width] * kernel[kernel_offset+1];
            center +=  kernel[kernel_offset+1];
        }
        if(is_not_bottom)
        {
            result[index] += image[index+index_offset+width] * kernel[kernel_offset+7];
            center +=  kernel[kernel_offset+7];
        }
        if(is_not_right)
        {
            result[index] += image[index+index_offset+1] * kernel[kernel_offset+5];
            center +=  kernel[kernel_offset+5];
            if(is_not_top)
            {
                result[index] += image[index+index_offset+1-width] * kernel[kernel_offset+2];
                center +=  kernel[kernel_offset+2];
            }
            if(is_not_bottom)
            {
                result[index] += image[index+index_offset+1+width] * kernel[kernel_offset+8];
                center +=  kernel[kernel_offset+8];
            }
        }
    }

    if(is_not_back)
    {
        const uint index_offset = width*height;
        const uint kernel_offset = 18;
        result[index] += image[index+index_offset] * kernel[kernel_offset+4];
        center +=  kernel[kernel_offset+4];
        if(is_not_left)
        {
            result[index] += image[index+index_offset-1] * kernel[kernel_offset+3];
            center +=  kernel[kernel_offset+3];
            if(is_not_top)
            {
                result[index] += image[index+index_offset-1-width] * kernel[kernel_offset];
                center +=  kernel[kernel_offset];
            }
            if(is_not_bottom)
            {
                result[index] += image[index+index_offset-1+width] * kernel[kernel_offset+6];
                center +=  kernel[kernel_offset+6];
            }
        }
        if(is_not_top)
        {
            result[index] += image[index+index_offset-width] * kernel[kernel_offset+1];
            center +=  kernel[kernel_offset+1];
        }
        if(is_not_bottom)
        {
            result[index] += image[index+index_offset+width] * kernel[kernel_offset+7];
            center +=  kernel[kernel_offset+7];
        }
        if(is_not_right)
        {
            result[index] += image[index+index_offset+1] * kernel[kernel_offset+5];
            center +=  kernel[kernel_offset+5];
            if(is_not_top)
            {
                result[index] += image[index+index_offset+1-width] * kernel[kernel_offset+2];
                center +=  kernel[kernel_offset+2];
            }
            if(is_not_bottom)
            {
                result[index] += image[index+index_offset+1+width] * kernel[kernel_offset+8];
                center +=  kernel[kernel_offset+8];
            }
        }
    }

    // center layer
    const uint kernel_offset = 9;

    if(is_not_left)
    {
        result[index] += image[index-1] * kernel[kernel_offset+3];
        center +=  kernel[kernel_offset+3];
        if(is_not_top)
        {
            result[index] += image[index-1-width] * kernel[kernel_offset];
            center +=  kernel[kernel_offset];
        }
        if(is_not_bottom)
        {
            result[index] += image[index-1+width] * kernel[kernel_offset+6];
            center +=  kernel[kernel_offset+6];
        }
    }
    if(is_not_top)
    {
        result[index] += image[index-width] * kernel[kernel_offset+1];
        center +=  kernel[kernel_offset+1];
    }
    if(is_not_bottom)
    {
        result[index] += image[index+width] * kernel[kernel_offset+7];
        center +=  kernel[kernel_offset+7];
    }
    if(is_not_right)
    {
        result[index] += image[index+1] * kernel[kernel_offset+5];
        center +=  kernel[kernel_offset+5];
        if(is_not_top)
        {
            result[index] += image[index+1-width] * kernel[kernel_offset+2];
            center +=  kernel[kernel_offset+2];
        }
        if(is_not_bottom)
        {
            result[index] += image[index+1+width] * kernel[kernel_offset+8];
            center +=  kernel[kernel_offset+8];
        }
    }

    result[index] -= image[index] * center;
}

template<typename Pixel>
Pixel* convolution3x3x3_kernel_launch(Pixel* image_host,
                              uint width, uint height, uint depth,
                                    Pixel* kernel_host, bool correct_center)
{
    int cuda_device_count;
    cudaCheckError( cudaGetDeviceCount(&cuda_device_count) );

    printf("found %d cuda devices.\n", cuda_device_count);

    uint voxel_count = width*height*depth;
    dim3 block_dimension(CUDA_BLOCK_DIMENSON);
    dim3 grid_dimension((voxel_count + block_dimension.x - 1) / block_dimension.x);

    Pixel* image, *kernel, *result;

    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMallocManaged(&image, size) )
    cudaCheckError( cudaMemcpy(image, image_host, size, cudaMemcpyHostToDevice) )

    size_t kernel_size = sizeof(Pixel) * 27;
    cudaCheckError( cudaMallocManaged(&kernel, size) )
    cudaCheckError( cudaMemcpy(kernel, kernel_host, kernel_size, cudaMemcpyHostToDevice) )

    cudaCheckError( cudaMallocManaged(&result, size) )

    cudaCheckError( cudaDeviceSynchronize() );

    if(correct_center)
        convolution3x3x3_calculate_center_kernel<<<grid_dimension, block_dimension>>>(
          image, width, height, depth,
          kernel, result);
    else
        convolution3x3x3_kernel<<<grid_dimension, block_dimension>>>(
          image, width, height, depth,
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

template float* convolution3x3x3_kernel_launch(float* image_host,
                              uint width, uint height, uint depth,
                              float* kernel_host, bool correct_center);
template double* convolution3x3x3_kernel_launch(double* image_host,
                              uint width, uint height, uint depth,
                              double* kernel_host, bool correct_center);
