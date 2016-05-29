#include "tgv_common.cu"

template<typename Pixel>
__global__ void multiply_kernel(
        Pixel* image1, Pixel* image2,
        const uint width, const uint height, const uint depth)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    image1[index] = image1[index] * image2[index];
}

template<typename Pixel>
Pixel* multiply_kernel_launch(Pixel* image1_host, Pixel* image2_host,
                  uint width, uint height, uint depth)
{
    int cuda_device_count;
    cudaCheckError( cudaGetDeviceCount(&cuda_device_count) );

    printf("found %d cuda devices.\n", cuda_device_count);

    uint voxel_count = width*height*depth;
    dim3 block_dimension(CUDA_BLOCK_DIMENSON);
    dim3 grid_dimension((voxel_count + block_dimension.x - 1) / block_dimension.x);

    Pixel* image1, *image2;
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMallocManaged(&image1, size) )
    cudaCheckError( cudaMemcpy(image1, image1_host, size, cudaMemcpyHostToDevice) )
    cudaCheckError( cudaMallocManaged(&image2, size) )
    cudaCheckError( cudaMemcpy(image2, image2_host, size, cudaMemcpyHostToDevice) )

    multiply_kernel<<<grid_dimension, block_dimension>>>(
         image1, image2, width, height, depth);

    Pixel* result = new Pixel[voxel_count];

    cudaCheckError( cudaMemcpy(result, image1, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaDeviceSynchronize() );

    cudaCheckError( cudaFree(image1) );
    cudaCheckError( cudaFree(image2) );

    return result;
}

template float* multiply_kernel_launch(float* image1, float* image2,
                  uint width, uint height, uint depth);
template double* multiply_kernel_launch(double* image1, double* image2,
                  uint width, uint height, uint depth);
