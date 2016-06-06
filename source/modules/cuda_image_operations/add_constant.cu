
#include "cuda_helper.cuh"

template<typename Pixel>
__global__  void add_constant_kernel(Pixel* image,
                              uint width, uint height, uint depth,
                              Pixel constant, Pixel* result)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    result[index] = image[index] + constant;
}

template<typename Pixel>
Pixel* add_constant_kernel_launch(Pixel* image_host,
                              uint width, uint height, uint depth,
                              Pixel constant_host)
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


    add_constant_kernel<<<grid_dimension, block_dimension>>>(
      image, width, height, depth,
      constant_host, result);
    cudaCheckError( cudaDeviceSynchronize() );

    Pixel* result_host = new Pixel[voxel_count];
    cudaCheckError( cudaMemcpy(result_host, result, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaDeviceSynchronize() );

    cudaFree(image);
    cudaFree(result);

    return result_host;
}

template float* add_constant_kernel_launch(float* image_host,
                              uint width, uint height, uint depth,
                              float constant_host);
template double* add_constant_kernel_launch(double* image_host,
                              uint width, uint height, uint depth,
                              double constant_host);
