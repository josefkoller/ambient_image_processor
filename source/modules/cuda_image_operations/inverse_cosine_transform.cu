

#include "cuda_helper.cuh"

#define pi 3.14159265359

template<typename Pixel>
__global__  void inverse_cosine_transform_kernel(Pixel* image,
                              uint width, uint height, uint depth,
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

    result[index] = 0;
    for(uint z2 = 0; z2 < depth; z2++)
    {
        for(uint y2 = 0; y2 < height; y2++)
        {
            for(uint x2 = 0; x2 < width; x2++)
            {
                uint index2 = z2 * width*height + x2 + y2 * width;
                result[index] += image[index2]
                        * cosf(x* x2 * pi / (width-1))
                        * cosf(y * y2 * pi / (height-1))
                        * cosf(z * z2 * pi / (depth-1));
            }
        }
    }
}

template<typename Pixel>
__global__  void inverse_cosine_transform_kernel_2D(Pixel* image,
                              uint width, uint height,
                              Pixel* result)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height)
        return;

    const int y = floorf(index / width);
    const int x = index - y * width;

    result[index] = 0;
    for(uint y2 = 0; y2 < height; y2++)
    {
        for(uint x2 = 0; x2 < width; x2++)
        {
            uint index2 = x2 + y2 * width;
            result[index] += image[index2]
                    * cosf(x* x2 * pi / (width-1))
                    * cosf(y * y2 * pi / (height-1));
        }
    }
}

template<typename Pixel>
__global__  void zero_kernel(Pixel* image,
                              uint width, uint height)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height)
        return;

    image[index] = 0;
}

template<typename Pixel>
__global__  void inverse_cosine_transform_kernel_x(Pixel* image,
                              uint width, uint height,
                              Pixel* result)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height)
        return;

    const int y = floorf(index / width);
    const int x = index - y * width;

    const Pixel size = width;

    for(uint x2 = 1; x2 < width; x2++)
    {
        uint index2 = x2 + y * width;
        result[index] += image[index2]
                * cospi((x2 * (x+1)*0.5) / size);
    }
    result[index] *= 2;
    result[index] += image[y * width];
}

template<typename Pixel>
__global__  void inverse_cosine_transform_kernel_y(Pixel* image,
                              uint width, uint height,
                              Pixel* result)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height)
        return;

    const int y = floorf(index / width);
    const int x = index - y * width;

    const Pixel size = height;

    for(uint y2 = 1; y2 < height; y2++)
    {
        uint index2 = x + y2 * width;
        result[index] += image[index2]
                * cospi((y2 * (y+1)*0.5) / size);
    }
    result[index] *= 2;
    result[index] += image[x];
}

template<typename Pixel>
Pixel* inverse_cosine_transform_kernel_launch(Pixel* image_host,
                              uint width, uint height, uint depth)
{
    int cuda_device_count;
    cudaCheckError( cudaGetDeviceCount(&cuda_device_count) );

    printf("found %d cuda devices.\n", cuda_device_count);

    uint voxel_count = width*height*depth;
    dim3 block_dimension(CUDA_BLOCK_DIMENSON);
    dim3 grid_dimension((voxel_count + block_dimension.x - 1) / block_dimension.x);

    Pixel* image, *result;

    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMallocManaged(&image, size) )
    cudaCheckError( cudaMemcpy(image, image_host, size, cudaMemcpyHostToDevice) )

    cudaCheckError( cudaMallocManaged(&result, size) )
    cudaCheckError( cudaDeviceSynchronize() );

    cudaCheckError( cudaDeviceSynchronize() );

    if(depth == 1)
    {
        zero_kernel<<<grid_dimension, block_dimension>>>(result, width, height);
        cudaCheckError( cudaDeviceSynchronize() );

        inverse_cosine_transform_kernel_x<<<grid_dimension, block_dimension>>>(
          image, width, height,
          result);
        cudaCheckError( cudaDeviceSynchronize() );

        cudaCheckError( cudaMemcpy(image, result, size, cudaMemcpyDeviceToDevice) )
        cudaCheckError( cudaDeviceSynchronize() );

        inverse_cosine_transform_kernel_y<<<grid_dimension, block_dimension>>>(
          image, width, height,
          result);
        cudaCheckError( cudaDeviceSynchronize() );
    }
    else
    {
        inverse_cosine_transform_kernel<<<grid_dimension, block_dimension>>>(
          image, width, height, depth,
          result);
        cudaCheckError( cudaDeviceSynchronize() );
    }

    Pixel* result_host = new Pixel[voxel_count];
    cudaCheckError( cudaMemcpy(result_host, result, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaDeviceSynchronize() );

    cudaFree(image);
    cudaFree(result);

    return result_host;
}

template float* inverse_cosine_transform_kernel_launch(float* image_host,
                              uint width, uint height, uint depth);
template double* inverse_cosine_transform_kernel_launch(double* image_host,
                              uint width, uint height, uint depth);
