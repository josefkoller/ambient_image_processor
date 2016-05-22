

#include "cuda.h"

#include <stdio.h>
#include <functional>

template<typename Pixel>
using IterationCallback = std::function<void(uint iteration_index, uint iteration_count, Pixel* u)>;


#include "cuda_helper.cuh"


template<typename Pixel>
__global__ void clone2(
        Pixel* f, Pixel* u, Pixel* u_bar,
        uint width, uint height, uint depth) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width*height*depth)
        return;

    u[index] = u_bar[index] = f[index];
}

template<typename Pixel>
__global__ void zeroInit(
        Pixel* p_x, Pixel* p_y, Pixel* p_z,
        Pixel* p_xx, Pixel* p_yy, Pixel* p_zz,
        uint width, uint height, uint depth) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width*height*depth)
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
__global__ void tgv_kernel_part4(
        Pixel* p_x, Pixel* p_y, Pixel* p_z,
        Pixel* p_xx, Pixel* p_yy, Pixel* p_zz,
        const Pixel tau, Pixel* u, Pixel* f,
        const Pixel lambda,
        Pixel* u_previous, const Pixel theta, Pixel* u_bar,
        const uint width, const uint height, const uint depth)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    if(depth > 1)
        u[index] -= tau * (p_x[index] + p_y[index] + p_z[index]);
    else
        u[index] -= tau * (p_x[index] + p_y[index]);

    u[index] = (u[index] + tau*lambda*f[index]) / (1 + tau*lambda);

    u_bar[index] = u[index] + theta*(u[index] - u_previous[index]);

    /*
     *  Matlab Code:
          u = u - tau * nabla_t * p;
          u = (u + tau * lambda .* f)/(1 + tau * lambda);

          % overrelaxation
          u_bar = u + theta*(u - u_old);
    */
}

template<typename Pixel>
Pixel* tgv_launch(Pixel* f_host,
                  uint width, uint height, uint depth,
                  Pixel lambda,
                  uint iteration_count,
                  uint paint_iteration_interval,
                  IterationCallback<Pixel> iteration_finished_callback,
                  Pixel alpha0,
                  Pixel alpha1)
{
    int cuda_device_count;
    cudaCheckError( cudaGetDeviceCount(&cuda_device_count) );

    printf("found %d cuda devices.\n", cuda_device_count);

    const uint voxel_count = width*height*depth;

    const dim3 block_dimension(CUDA_BLOCK_DIMENSON);
    const dim3 grid_dimension(
                (voxel_count + block_dimension.x - 1) / block_dimension.x);

    const dim3 grid_dimension_x(
                (depth*height + block_dimension.x - 1) / block_dimension.x);
    const dim3 grid_dimension_y(
                (depth*width + block_dimension.x - 1) / block_dimension.x);
    const dim3 grid_dimension_z(
                (width*height + block_dimension.x - 1) / block_dimension.x);

    printf("block dimensions: x:%d, y:%d \n", block_dimension.x);
    printf("grid dimensions: x:%d, y:%d \n", grid_dimension.x);

    Pixel* f, *u;

    // algorithm variables..
    const Pixel sqrt_8 = std::sqrt(8.0);
    const Pixel tau = 1.0 / sqrt_8;
    const Pixel sigma = tau;
    const Pixel theta = 1;
    Pixel* u_previous, *u_bar, *p_x, *p_y, *p_z, *p_xx, *p_yy, *p_zz;

    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMallocManaged(&f, size) );
    cudaCheckError( cudaMemcpy(f, f_host, size, cudaMemcpyHostToDevice) );

    cudaCheckError( cudaMallocManaged(&u, size) );
    cudaCheckError( cudaMallocManaged(&u_previous, size) );
    cudaCheckError( cudaMallocManaged(&u_bar, size) );
    cudaCheckError( cudaMallocManaged(&p_x, size) );
    cudaCheckError( cudaMallocManaged(&p_y, size) );
    cudaCheckError( cudaMallocManaged(&p_xx, size) );
    cudaCheckError( cudaMallocManaged(&p_yy, size) );
    if(depth > 1) {
        cudaCheckError( cudaMallocManaged(&p_z, size) );
        cudaCheckError( cudaMallocManaged(&p_zz, size) );
    }

    // algorithm begin
    zeroInit<<<grid_dimension, block_dimension>>>(
                                                    p_x, p_y, p_z,
                                                    p_xx, p_yy, p_zz,
                                                    width, height, depth);
    clone2<<<grid_dimension, block_dimension>>>(
                                                  f, u, u_bar, width, height, depth);
    cudaCheckError( cudaDeviceSynchronize() );

    for(uint iteration_index = 0; iteration_index < iteration_count; iteration_index++)
    {
        /* matlab primal dual TVL2
          % dual update
          p = p + sigma*nabla*u_bar;
          norm_p = sqrt(p(1:N).^2 + p(N+1:2*N).^2);
          p = p./max(1,[norm_p; norm_p]);

          u_old = u;

          % primal update
          u = u - tau * nabla_t * p;
          u = (u + tau * lambda .* f)/(1 + tau * lambda);

          % overrelaxation
          u_bar = u + theta*(u - u_old);
      */

        forward_difference_x<<<grid_dimension_x, block_dimension>>>(
              u_bar, p_x, width, height, depth);
        forward_difference_y<<<grid_dimension_y, block_dimension>>>(
              u_bar, p_y, width, height, depth);
        if(depth > 1)
            forward_difference_z<<<grid_dimension_z, block_dimension>>>(
                 u_bar, p_z, width, height, depth);
        cudaCheckError( cudaDeviceSynchronize() );

        tgv_kernel_part2<<<grid_dimension, block_dimension>>>(
                                                                p_x, p_y, p_z,
                                                                p_xx, p_yy, p_zz,
                                                                sigma, alpha1, u_previous, u,
                                                                width, height, depth);
        cudaCheckError( cudaDeviceSynchronize() );

        backward_difference_x<<<grid_dimension_x, block_dimension>>>(
                                                                       p_xx, p_x, width, height, depth);
        backward_difference_y<<<grid_dimension_y, block_dimension>>>(
                                                                       p_yy, p_y, width, height, depth);
        if(depth > 1)
            backward_difference_z<<<grid_dimension_z, block_dimension>>>(
                                                                       p_zz, p_z, width, height, depth);
        cudaCheckError( cudaDeviceSynchronize() );

        tgv_kernel_part4<<<grid_dimension, block_dimension>>>(  p_x, p_y, p_z,
                                                                p_xx, p_yy, p_zz,
                                                                tau, u, f,
                                                                lambda,
                                                                u_previous, theta, u_bar,
                                                                width, height, depth);
        cudaCheckError( cudaDeviceSynchronize() );

        printf("TVL2, iteration=%d / %d \n", iteration_index, iteration_count);
        if(paint_iteration_interval > 0 &&
                iteration_index % paint_iteration_interval == 0) {
            iteration_finished_callback(iteration_index, iteration_count, u);
        }
    }
    Pixel* destination = new Pixel[voxel_count];
    cudaCheckError( cudaMemcpy(destination, u, size, cudaMemcpyDeviceToHost) );
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

    return destination;
}

// generate the algorithm explicitly for...

template float* tgv_launch(float* f_host,
uint width, uint height, uint depth,
float lambda,
uint iteration_count,
uint paint_iteration_interval,
IterationCallback<float> iteration_finished_callback,
float alpha0,
float alpha1);

template double* tgv_launch(double* f_host,
uint width, uint height, uint depth,
double lambda,
uint iteration_count,
uint paint_iteration_interval,
IterationCallback<double> iteration_finished_callback,
double alpha0,
double alpha1);
