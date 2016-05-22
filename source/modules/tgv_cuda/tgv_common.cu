

#include "cuda.h"

#include <functional>

template<typename Pixel>
using IterationCallback = std::function<void(uint iteration_index, uint iteration_count, Pixel* u)>;

#include "cuda_helper.cuh"


template<typename Pixel>
__global__ void clone2(
        Pixel* f, Pixel* u, Pixel* u_bar,
        uint voxel_count) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= voxel_count)
        return;

    u[index] = u_bar[index] = f[index];
}

template<typename Pixel>
__global__ void zeroInit(
        Pixel* p_x, Pixel* p_y, Pixel* p_z,
        Pixel* p_xx, Pixel* p_yy, Pixel* p_zz,
        uint voxel_count, uint depth) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= voxel_count)
        return;

    p_x[index] = p_y[index] =
    p_xx[index] = p_yy[index] = 0;

    if(depth > 1)
        p_z[index] = p_zz[index] = 0;
}

template<typename Pixel>
__global__ void forward_difference_x(
        Pixel* u_bar, Pixel* p_x, const uint width, const uint height, const uint depth) {

    const uint index1 = blockDim.x * blockIdx.x + threadIdx.x;

    if(index1 >= height*depth)
        return;

    const uint y = floorf(index1 / depth);
    const uint z = index1 - y*depth;

    const uint offset = z*width*height + y*width;

    p_x[offset + width - 1] = 0; // neumann boundary condition
    for(uint x = 0; x < width - 1; x++)
    {
        const uint index2 = offset + x;
        p_x[index2] = u_bar[index2 + 1] - u_bar[index2];
    }
}

template<typename Pixel>
__global__ void forward_difference_y(
        Pixel* u_bar, Pixel* p_y, const uint width, const uint height, const uint depth) {

    const uint index1 = blockDim.x * blockIdx.x + threadIdx.x;

    if(index1 >= width*depth)
        return;

    const uint x = floorf(index1 / depth);
    const uint z = index1 - x*depth;

    const uint offset = z*width*height + x;

    p_y[offset + (height - 1) * width] = 0; // neumann boundary condition
    for(uint y = 0; y < height - 1; y++)
    {
        const uint index2 = offset + y * width;
        p_y[index2] = u_bar[index2 + width] - u_bar[index2];
    }
}

template<typename Pixel>
__global__ void forward_difference_z(
        Pixel* u_bar, Pixel* p_z, const uint width, const uint height, const uint depth) {

    const uint index1 = blockDim.x * blockIdx.x + threadIdx.x;

    const uint width_x_height = width * height;
    if(index1 >= width_x_height)
        return;

    const uint y = floorf(index1 / width);
    const uint x = index1 - y*width;

    const uint offset = y*width + x;

    p_z[offset + (depth - 1) * width_x_height] = 0; // neumann boundary condition
    for(uint z = 0; z < depth - 1; z++)
    {
        const uint index2 = offset + z * width_x_height;
        p_z[index2] = u_bar[index2 + width_x_height] - u_bar[index2];
    }
}

template<typename Pixel>
__global__ void backward_difference_x(
        Pixel* u_bar, Pixel* p_x, const uint width, const uint height, const uint depth) {

    const uint index1 = blockDim.x * blockIdx.x + threadIdx.x;

    if(index1 >= height*depth)
        return;

    const uint y = floorf(index1 / depth);
    const uint z = index1 - y*depth;

    const uint offset = z*width*height + y*width;

    p_x[offset] = - u_bar[offset]; // neumann boundary condition of gradient
    for(uint x = 1; x < width; x++)
    {
        const uint index2 = offset + x;
        p_x[index2] = - u_bar[index2] + u_bar[index2 - 1];  // note: the sign
    }
}

template<typename Pixel>
__global__ void backward_difference_y(
        Pixel* u_bar, Pixel* p_y, const uint width, const uint height, const uint depth) {

    const uint index1 = blockDim.x * blockIdx.x + threadIdx.x;

    if(index1 >= width*depth)
        return;

    const uint x = floorf(index1 / depth);
    const uint z = index1 - x*depth;

    const uint offset = z*width*height + x;

    p_y[offset] = - u_bar[offset]; // neumann boundary condition
    for(uint y = 1; y < height; y++)
    {
        const uint index2 = offset + y * width;
        p_y[index2] = - u_bar[index2] + u_bar[index2 - width] ;
    }
}

template<typename Pixel>
__global__ void backward_difference_z(
        Pixel* u_bar, Pixel* p_z, const uint width, const uint height, const uint depth) {

    const uint index1 = blockDim.x * blockIdx.x + threadIdx.x;

    const uint width_x_height = width * height;
    if(index1 >= width_x_height)
        return;

    const uint y = floorf(index1 / width);
    const uint x = index1 - y*width;

    const uint offset = y*width + x;

    p_z[offset] = - u_bar[offset]; // neumann boundary condition
    for(uint z = 1; z < depth; z++)
    {
        const uint index2 = offset + z * width_x_height;
        p_z[index2] = - u_bar[index2] + u_bar[index2 - width_x_height];
    }
}



template<typename Pixel>
__global__ void tgv_kernel_part2(
        Pixel* p_x, Pixel* p_y, Pixel* p_z,
        Pixel* p_xx, Pixel* p_yy, Pixel* p_zz,
        const Pixel sigma, const Pixel alpha1, Pixel* u_previous, Pixel* u,
        const uint width, const uint height, const uint depth) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    /*
    * Matlab Code:
    p = p + sigma*nabla*u_bar;
    norm_p = sqrt(p(1:N).^2 + p(N+1:2*N).^2);
    p = p./max(1,[norm_p; norm_p]);

    u_old = u;
    */

    p_xx[index] += sigma * p_x[index];
    p_yy[index] += sigma * p_y[index];
    if(depth > 1)
        p_zz[index] += sigma * p_z[index];

    Pixel normalization =
            p_xx[index] * p_xx[index] +
            p_yy[index] * p_yy[index];
    if(depth > 1)
        normalization += p_zz[index] * p_zz[index];

    normalization = fmax(alpha1, sqrt(normalization));

    p_xx[index] /= normalization;
    p_yy[index] /= normalization;
    if(depth > 1)
        p_zz[index] /= normalization;

    u_previous[index] = u[index];
}


template<typename Pixel>
void tgv_launch_part1(
          uint width, uint height, uint depth,
          uint &voxel_count,
          dim3 &block_dimension,
          dim3 &grid_dimension,
          dim3 &grid_dimension_x,
          dim3 &grid_dimension_y,
          dim3 &grid_dimension_z)
{
    int cuda_device_count;
    cudaCheckError( cudaGetDeviceCount(&cuda_device_count) );

    printf("found %d cuda devices.\n", cuda_device_count);

    voxel_count = width*height*depth;

    block_dimension = dim3(CUDA_BLOCK_DIMENSON);
    grid_dimension = dim3((voxel_count + block_dimension.x - 1) / block_dimension.x);

    grid_dimension_x = dim3((depth*height + block_dimension.x - 1) / block_dimension.x);
    grid_dimension_y = dim3((depth*width + block_dimension.x - 1) / block_dimension.x);
    grid_dimension_z = dim3((width*height + block_dimension.x - 1) / block_dimension.x);

    printf("block dimensions: x:%d, y:%d \n", block_dimension.x);
    printf("grid dimensions: x:%d, y:%d \n", grid_dimension.x);
}

template<typename Pixel>
void tgv_launch_part2(Pixel* f_host,
          uint voxel_count, uint depth,
          Pixel** f, Pixel** u,
          Pixel** u_previous, Pixel**u_bar,
          Pixel**p_x, Pixel**p_y, Pixel**p_z,
          Pixel**p_xx, Pixel**p_yy, Pixel**p_zz) {

    printf("voxel_count: %d \n", voxel_count);

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
    if(depth > 1) {
        cudaCheckError( cudaMallocManaged(p_z, size) )
        cudaCheckError( cudaMallocManaged(p_zz, size) )
    }
}

template<typename Pixel>
void tgv_launch_forward_differences(Pixel* u_bar,
        Pixel* p_x, Pixel* p_y, Pixel* p_z,
        uint width, uint height, uint depth,
        dim3 block_dimension,
        dim3 grid_dimension_x,
        dim3 grid_dimension_y,
        dim3 grid_dimension_z)
{
    forward_difference_x<<<grid_dimension_x, block_dimension>>>(
          u_bar, p_x, width, height, depth);
    forward_difference_y<<<grid_dimension_y, block_dimension>>>(
          u_bar, p_y, width, height, depth);
    if(depth > 1)
        forward_difference_z<<<grid_dimension_z, block_dimension>>>(
             u_bar, p_z, width, height, depth);
    cudaCheckError( cudaDeviceSynchronize() );
}

template<typename Pixel>
void tgv_launch_backward_differences(
        Pixel* p_x, Pixel* p_y, Pixel* p_z,
        Pixel* p_xx, Pixel* p_yy, Pixel* p_zz,
        uint width, uint height, uint depth,
        dim3 block_dimension,
        dim3 grid_dimension_x,
        dim3 grid_dimension_y,
        dim3 grid_dimension_z)
{
    backward_difference_x<<<grid_dimension_x, block_dimension>>>(
                                                                   p_xx, p_x, width, height, depth);
    backward_difference_y<<<grid_dimension_y, block_dimension>>>(
                                                                   p_yy, p_y, width, height, depth);
    if(depth > 1)
        backward_difference_z<<<grid_dimension_z, block_dimension>>>(
                                                                   p_zz, p_z, width, height, depth);
    cudaCheckError( cudaDeviceSynchronize() );
}


template<typename Pixel>
void tgv_launch_part3(
            Pixel* host_f,
            uint voxel_count, uint depth,
            Pixel* u_previous, Pixel* u_bar,
            Pixel* p_x, Pixel* p_y, Pixel* p_z,
            Pixel* p_xx, Pixel* p_yy, Pixel* p_zz,
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
    if(depth > 1) {
        cudaFree(p_z);
        cudaFree(p_zz);
    }
    cudaFree(f);
    cudaFree(u);
}
