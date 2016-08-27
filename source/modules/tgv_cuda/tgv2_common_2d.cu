#ifndef TGV_2_COMMON_2D
#define TGV_2_COMMON_2D

#include "cuda_helper.cuh"

#include "tgv_common_2d.cu"

template<typename Pixel>
__global__ void addAndHalf_2d(
        Pixel* v_xy, Pixel* v_yx, Pixel* q_xy,
        uint width, uint height) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width*height)
        return;

    q_xy[index] = (v_xy[index] + v_yx[index]) * 0.5;
}
template<typename Pixel>
__global__ void add_2d(
        Pixel* v_xy, Pixel* v_yx, Pixel* q_xy,
        uint width, uint height) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width*height)
        return;

    q_xy[index] = v_xy[index] + v_yx[index];
}


template<typename Pixel>
void tgv_launch_part22_2d(
        uint voxel_count,
        Pixel** v_bar_x, Pixel** v_bar_y,
        Pixel** v_previous_x, Pixel** v_previous_y,
        Pixel** v_x, Pixel** v_y,
        Pixel** q_x, Pixel** q_y,
        Pixel** q_xy,
        Pixel** q_x2, Pixel** q_y2,
        Pixel** q_xy2, Pixel** q_temp)
{
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMallocManaged(v_x, size) )
    cudaCheckError( cudaMallocManaged(v_y, size) )
    cudaCheckError( cudaMallocManaged(v_bar_x, size) )
    cudaCheckError( cudaMallocManaged(v_bar_y, size) )
    cudaCheckError( cudaMallocManaged(v_previous_x, size) )
    cudaCheckError( cudaMallocManaged(v_previous_y, size) )

    cudaCheckError( cudaMallocManaged(q_x, size) )
    cudaCheckError( cudaMallocManaged(q_y, size) )
    cudaCheckError( cudaMallocManaged(q_xy, size) )
    cudaCheckError( cudaMallocManaged(q_x2, size) )
    cudaCheckError( cudaMallocManaged(q_y2, size) )
    cudaCheckError( cudaMallocManaged(q_xy2, size) )
    cudaCheckError( cudaMallocManaged(q_temp, size) )
}

template<typename Pixel>
__global__ void tgv_kernel_part22_2d(
        Pixel* v_bar_x, Pixel* v_bar_y,
        Pixel* p_x, Pixel* p_y,
        Pixel* p_xx, Pixel* p_yy,
        const Pixel sigma, const Pixel alpha1, Pixel* u_previous, Pixel* u,
        const uint width, const uint height) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height)
        return;

    /*
    * Matlab Code:
    p = p + sigma*(nabla*u_bar - v_bar);
    norm_p  = sqrt(p(1:N).^2 + p(N+1:2*N).^2 +  p(2*N+1:3*N).^2);
    p = p./max(1,[norm_p; norm_p; norm_p]/alpha1);

    u_old = u;
    */


    p_xx[index] = fmaf(sigma, p_x[index] - v_bar_x[index], p_xx[index]);
    p_yy[index] = fmaf(sigma, p_y[index] - v_bar_y[index], p_yy[index]);

    /*
    /// TGV1
    p_xx[index] += sigma * (p_x[index]);
    p_yy[index] += sigma * (p_y[index]);
    if(depth > 1)
        p_zz[index] += sigma * (p_z[index]);
    */

    Pixel normalization = sqrtf(p_xx[index] * p_xx[index] + p_yy[index] * p_yy[index]);

    normalization = fmaxf(1, normalization/alpha1);

    p_xx[index] /= normalization;
    p_yy[index] /= normalization;

    u_previous[index] = u[index];
}

template<typename Pixel>
void tgv_launch_part32_2d(
                       Pixel* v_bar_x, Pixel* v_bar_y,
                       Pixel* v_previous_x, Pixel* v_previous_y,
                       Pixel* v_x, Pixel* v_y,
                       Pixel* q_x, Pixel* q_y,
                       Pixel* q_xy,
                       Pixel* q_x2, Pixel* q_y2,
                       Pixel* q_xy2, Pixel* q_temp)
{
    cudaFree(v_x);
    cudaFree(v_y);
    cudaFree(v_bar_x);
    cudaFree(v_bar_y);
    cudaFree(v_previous_x);
    cudaFree(v_previous_y);

    cudaFree(q_x);
    cudaFree(q_y);
    cudaFree(q_xy);
    cudaFree(q_x2);
    cudaFree(q_y2);
    cudaFree(q_xy2);
    cudaFree(q_temp);
}



template<typename Pixel>
void tgv_launch_gradient2_2d(
        Pixel* v_bar_x, Pixel* v_bar_y,
        Pixel* q_x, Pixel* q_y,
        Pixel* q_xy,  Pixel* q_temp,
        uint width, uint height,
        dim3 block_dimension,
        dim3 grid_dimension,
        dim3 grid_dimension_x,
        dim3 grid_dimension_y)
{
    backward_difference_x_2d<<<grid_dimension_x, block_dimension>>>(
          v_bar_x, q_x, width, height);

    backward_difference_y_2d<<<grid_dimension_y, block_dimension>>>(
          v_bar_y, q_y, width, height);

    backward_difference_x_2d<<<grid_dimension_x, block_dimension>>>(
          v_bar_y, q_xy, width, height);
    backward_difference_y_2d<<<grid_dimension_y, block_dimension>>>(
          v_bar_x, q_temp, width, height);
    addAndHalf_2d<<<grid_dimension, block_dimension>>>(
            q_xy, q_temp, q_xy,
            width, height);

    cudaCheckError( cudaDeviceSynchronize() );
}

template<typename Pixel>
__global__ void tgv_kernel_part5_2d(
        Pixel* v_x,Pixel* v_y,
        Pixel* v_previous_x, Pixel* v_previous_y,
        Pixel* q_x,Pixel* q_y,
        Pixel* q_xy,
        Pixel* q_x2, Pixel* q_y2,
        Pixel* q_xy2,
        const Pixel sigma, const Pixel alpha0,
        const uint width, const uint height) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height)
        return;

       /*
        * Matlab Code:
        % dual update q
        q = q + sigma*nabla_second*v_bar;
        norm_q = sqrt(q(1:N).^2 + q(N+1:2*N).^2 + q(2*N+1:3*N).^2 + ... % main diagonal
            2*q(3*N+1:4*N).^2 + 2*q(4*N+1:5*N).^2 + 2*q(5*N+1:6*N).^2); % off diagonal
        q = q./max(1, repmat(norm_q, 6, 1)/alpha0);
       */

    q_x2[index] = fmaf(sigma, q_x[index], q_x2[index]);
    q_y2[index] = fmaf(sigma, q_y[index], q_y2[index]);
    q_xy2[index] = fmaf(sigma, q_xy[index], q_xy2[index]);

    Pixel normalization =
            q_x2[index] * q_x2[index] +
            q_y2[index] * q_y2[index] +
            2 * q_xy2[index] * q_xy2[index];

    normalization = fmaxf(1, sqrtf(normalization) / alpha0);

    q_x2[index] /= normalization;
    q_y2[index] /= normalization;
    q_xy2[index] /= normalization;

    v_previous_x[index] = v_x[index];
    v_previous_y[index] = v_y[index];
}

template<typename Pixel>
void tgv_launch_divergence2_2d(
        Pixel* q_x, Pixel* q_y,
        Pixel* q_xy,
        Pixel* q_x2, Pixel* q_y2,
        Pixel* q_temp,
        uint width, uint height,
        dim3 block_dimension,
        dim3 grid_dimension,
        dim3 grid_dimension_x,
        dim3 grid_dimension_y)
{
    forward_difference_x_2d<<<grid_dimension_x, block_dimension>>>(
      q_x, q_x2, width, height);
    forward_difference_y_2d<<<grid_dimension_y, block_dimension>>>(
      q_xy, q_temp, width, height);
    add_2d<<<grid_dimension, block_dimension>>>(q_x2, q_temp, q_x2, width, height);

    forward_difference_y_2d<<<grid_dimension_y, block_dimension>>>(
      q_y, q_y2, width, height);
    forward_difference_x_2d<<<grid_dimension_x, block_dimension>>>(
      q_xy, q_temp, width, height);
    add_2d<<<grid_dimension, block_dimension>>>(q_y2, q_temp, q_y2, width, height);

    cudaCheckError( cudaDeviceSynchronize() );
}


template<typename Pixel>
__global__ void tgv_kernel_part6_2d(
        Pixel*  v_x, Pixel*  v_y,
        Pixel* q_x2, Pixel* q_y2,
        Pixel* p_x, Pixel* p_y,
        Pixel* v_previous_x, Pixel* v_previous_y,
        Pixel* v_bar_x, Pixel* v_bar_y,
        const Pixel tau, const Pixel theta,
        const uint width, const uint height)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height)
        return;

    v_x[index] -= tau * (q_x2[index] - p_x[index]);
    v_y[index] -= tau * (q_y2[index] - p_y[index]);

    v_bar_x[index] = v_x[index] + theta*(v_x[index] - v_previous_x[index]);
    v_bar_y[index] = v_y[index] + theta*(v_y[index] - v_previous_y[index]);

    /*
     *  Matlab Code:
            v = v - tau * (nabla_second_t * q - p);
            v_bar = v + theta*(v - v_old);
    */
}

template<typename Pixel>
__global__ void zeroInit2_2d(
        Pixel* p_x, Pixel* p_y,
        Pixel* p_xx,
        uint voxel_count) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= voxel_count)
        return;

    p_x[index] = p_y[index] = p_xx[index] = 0;
}

#endif // TGV_2_COMMON_2D
