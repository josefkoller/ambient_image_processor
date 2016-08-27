#ifndef TGV_COMMON_2D
#define TGV_COMMON_2D


#include <functional>

#include "cuda_helper.cuh"

// clone2, IterationCallback
#include "tgv_common.cu"


template<typename Pixel>
__global__ void zeroInit_2d(
        Pixel* p_x, Pixel* p_y,
        Pixel* p_xx, Pixel* p_yy,
        uint voxel_count) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= voxel_count)
        return;

    p_x[index] = p_y[index] =
    p_xx[index] = p_yy[index] = 0;
}

template<typename Pixel>
__global__ void forward_difference_x_2d(
        Pixel* u_bar, Pixel* p_x, const uint width, const uint height) {

    const uint y = blockDim.x * blockIdx.x + threadIdx.x;

    if(y >= height)
        return;

    const uint offset = y*width;

    p_x[offset + width - 1] = 0; // neumann boundary condition
    for(uint x = 0; x < width - 1; x++)
    {
        const uint index2 = offset + x;
        p_x[index2] = u_bar[index2 + 1] - u_bar[index2];
    }
}

template<typename Pixel>
__global__ void forward_difference_y_2d(
        Pixel* u_bar, Pixel* p_y, const uint width, const uint height) {

    const uint x = blockDim.x * blockIdx.x + threadIdx.x;

    if(x >= width)
        return;

    p_y[x + (height - 1) * width] = 0; // neumann boundary condition
    for(uint y = 0; y < height - 1; y++)
    {
        const uint index2 = x + y * width;
        p_y[index2] = u_bar[index2 + width] - u_bar[index2];
    }
}

template<typename Pixel>
__global__ void backward_difference_x_2d(
        Pixel* u_bar, Pixel* p_x, const uint width, const uint height) {

    const uint y = blockDim.x * blockIdx.x + threadIdx.x;

    if(y >= height)
        return;

    const uint offset = y*width;

    p_x[offset] = - u_bar[offset]; // neumann boundary condition of gradient
    for(uint x = 1; x < width; x++)
    {
        const uint index2 = offset + x;
        p_x[index2] = - u_bar[index2] + u_bar[index2 - 1];  // note: the sign
    }
}

template<typename Pixel>
__global__ void backward_difference_y_2d(
        Pixel* u_bar, Pixel* p_y, const uint width, const uint height) {

    const uint x = blockDim.x * blockIdx.x + threadIdx.x;

    if(x >= width)
        return;

    p_y[x] = - u_bar[x]; // neumann boundary condition
    for(uint y = 1; y < height; y++)
    {
        const uint index2 = x + y * width;
        p_y[index2] = - u_bar[index2] + u_bar[index2 - width] ;
    }
}

template<typename Pixel>
__global__ void tgv_kernel_part2_2d(
        Pixel* p_x, Pixel* p_y,
        Pixel* p_xx, Pixel* p_yy,
        const Pixel sigma, const Pixel alpha1, Pixel* u_previous, Pixel* u,
        const uint width, const uint height) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height)
        return;

    /*
    * Matlab Code:
    p = p + sigma*nabla*u_bar;
    norm_p = sqrt(p(1:N).^2 + p(N+1:2*N).^2);
    p = p./max(1,[norm_p; norm_p]);

    u_old = u;
    */

    p_xx[index] = fmaf(sigma, p_x[index], p_xx[index]);
    p_yy[index] = fmaf(sigma, p_y[index], p_yy[index]);

    Pixel normalization = sqrtf(p_xx[index] * p_xx[index] + p_yy[index] * p_yy[index]);

    normalization = fmaxf(1, normalization/alpha1);

    p_xx[index] /= normalization;
    p_yy[index] /= normalization;

    u_previous[index] = u[index];
}


template<typename Pixel>
void tgv_launch_part1_2d(
          uint width, uint height,
          uint &voxel_count,
          dim3 &block_dimension,
          dim3 &grid_dimension,
          dim3 &grid_dimension_x,
          dim3 &grid_dimension_y)
{
    int cuda_device_count;
    cudaCheckError( cudaGetDeviceCount(&cuda_device_count) );

 //   printf("found %d cuda devices.\n", cuda_device_count);

    voxel_count = width*height;

    block_dimension = dim3(CUDA_BLOCK_DIMENSON);
    grid_dimension = dim3((voxel_count + block_dimension.x - 1) / block_dimension.x);

    grid_dimension_x = dim3((height + block_dimension.x - 1) / block_dimension.x);
    grid_dimension_y = dim3((width + block_dimension.x - 1) / block_dimension.x);

   // printf("block dimensions: x:%d \n", block_dimension.x);
  //  printf("grid dimensions: x:%d  \n", grid_dimension.x);
}

template<typename Pixel>
void tgv_launch_part2_2d(Pixel* f_host,
          uint voxel_count,
          Pixel** f, Pixel** u,
          Pixel** u_previous, Pixel** u_bar,
          Pixel** p_x, Pixel** p_y,
          Pixel** p_xx, Pixel** p_yy) {

//    printf("voxel_count: %d \n", voxel_count);

    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMallocManaged(f, size) )
    cudaCheckError( cudaMemcpy(*f, f_host, size, cudaMemcpyHostToDevice) )

    cudaCheckError( cudaMallocManaged(u, size) )
    cudaCheckError( cudaMallocManaged(u_previous, size) )
    cudaCheckError( cudaMallocManaged(u_bar, size) )
    cudaCheckError( cudaMallocManaged(p_x, size) )
    cudaCheckError( cudaMallocManaged(p_y, size) )
    cudaCheckError( cudaMallocManaged(p_xx, size) )
    cudaCheckError( cudaMallocManaged(p_yy, size) )
}


template<typename Pixel>
void tgv_launch_forward_differences_2d(Pixel* u_bar,
        Pixel* p_x, Pixel* p_y,
        uint width, uint height,
        dim3 block_dimension,
        dim3 grid_dimension_x,
        dim3 grid_dimension_y)
{
    forward_difference_x_2d<<<grid_dimension_x, block_dimension>>>(
          u_bar, p_x, width, height);
    forward_difference_y_2d<<<grid_dimension_y, block_dimension>>>(
          u_bar, p_y, width, height);
    cudaCheckError( cudaDeviceSynchronize() );
}

template<typename Pixel>
void tgv_launch_backward_differences_2d(
        Pixel* p_x, Pixel* p_y,
        Pixel* p_xx, Pixel* p_yy,
        uint width, uint height,
        dim3 block_dimension,
        dim3 grid_dimension_x,
        dim3 grid_dimension_y)
{
    backward_difference_x_2d<<<grid_dimension_x, block_dimension>>>(p_xx, p_x, width, height);
    backward_difference_y_2d<<<grid_dimension_y, block_dimension>>>(p_yy, p_y, width, height);
    cudaCheckError( cudaDeviceSynchronize() );
}


template<typename Pixel>
void tgv_launch_part3_2d(
            Pixel* host_f,
            uint voxel_count,
            Pixel* u_previous, Pixel* u_bar,
            Pixel* p_x, Pixel* p_y,
            Pixel* p_xx, Pixel* p_yy,
            Pixel* f, Pixel* u)
{
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMemcpy(host_f, u, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaDeviceSynchronize() );

    cudaFree(u_previous);
    cudaFree(u_bar);
    cudaFree(p_x);
    cudaFree(p_y);
    cudaFree(p_xx);
    cudaFree(p_yy);
    cudaFree(f);
    cudaFree(u);
}

#endif //TGV_COMMON_2D
