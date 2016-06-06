

#include "cuda_helper.cuh"

template<typename Pixel>
__global__  void solve_poisson_in_cosine_domain_kernel(Pixel* image,
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

    result[index] = image[index] /
            (4
             - 2*cospi(x / ((Pixel)width))
             - 2*cospi(y / ((Pixel)height))
             - 2*cospi(z / ((Pixel)depth)) );
}

template<typename Pixel>
Pixel* solve_poisson_in_cosine_domain_kernel_launch(Pixel* image_host,
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

    solve_poisson_in_cosine_domain_kernel<<<grid_dimension, block_dimension>>>(
      image, width, height, depth,
      result);
    cudaCheckError( cudaDeviceSynchronize() );

    Pixel* result_host = new Pixel[voxel_count];
    cudaCheckError( cudaMemcpy(result_host, result, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaDeviceSynchronize() );

    cudaFree(image);
    cudaFree(result);
    cudaCheckError( cudaDeviceSynchronize() );

    return result_host;
}

template float* solve_poisson_in_cosine_domain_kernel_launch(float* image_host,
                              uint width, uint height, uint depth);
template double* solve_poisson_in_cosine_domain_kernel_launch(double* image_host,
                              uint width, uint height, uint depth);
