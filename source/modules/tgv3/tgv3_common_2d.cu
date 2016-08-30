#ifndef TGV_3_COMMON_2D
#define TGV_3_COMMON_2D

#include "tgv2_common_2d.cu"

template<typename Pixel>
__global__ void tgv3_kernel_part5_2d(
        Pixel* v_x,Pixel* v_y,
        Pixel* v_previous_x, Pixel* v_previous_y,
        Pixel* q_x,Pixel* q_y,
        Pixel* q_xy,
        Pixel* q_x2, Pixel* q_y2,
        Pixel* q_xy2,

        Pixel* w_x,Pixel* w_y,
        Pixel* w_xy,

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

    q_x2[index] = fmaf(sigma, q_x[index] - w_x[index], q_x2[index]);
    q_y2[index] = fmaf(sigma, q_y[index] - w_y[index], q_y2[index]);
    q_xy2[index] = fmaf(sigma,  q_xy[index] - w_xy[index], q_xy2[index]);

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
void tgv3_launch_part23_2d(
        uint voxel_count,

        Pixel** w_x, Pixel** w_y,
        Pixel** w_xy,

        Pixel** w_bar_x, Pixel** w_bar_y,
        Pixel** w_bar_xy,

        Pixel** w_previous_x, Pixel** w_previous_y,
        Pixel** w_previous_xy,

        Pixel** r_x, Pixel** r_y,
        Pixel** r_xy,

        Pixel** r2_x, Pixel** r2_y,
        Pixel** r2_xy)
{
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMalloc(w_x, size) )
    cudaCheckError( cudaMalloc(w_bar_x, size) )
    cudaCheckError( cudaMalloc(w_previous_x, size) )

    cudaCheckError( cudaMalloc(w_y, size) )
    cudaCheckError( cudaMalloc(w_bar_y, size) )
    cudaCheckError( cudaMalloc(w_previous_y, size) )

    cudaCheckError( cudaMalloc(w_xy, size) )
    cudaCheckError( cudaMalloc(w_bar_xy, size) )
    cudaCheckError( cudaMalloc(w_previous_xy, size) )

    cudaCheckError( cudaMalloc(r_x, size) )
    cudaCheckError( cudaMalloc(r_y, size) )
    cudaCheckError( cudaMalloc(r_xy, size) )

    cudaCheckError( cudaMalloc(r2_x, size) )
    cudaCheckError( cudaMalloc(r2_y, size) )
    cudaCheckError( cudaMalloc(r2_xy, size) )
}


template<typename Pixel>
void tgv3_launch_part33_2d(
        Pixel* w_x, Pixel* w_y,
        Pixel* w_xy,

        Pixel* w_bar_x, Pixel* w_bar_y,
        Pixel* w_bar_xy,

        Pixel* w_previous_x, Pixel* w_previous_y,
        Pixel* w_previous_xy,

        Pixel* r_x, Pixel* r_y,
        Pixel* r_xy,

        Pixel* r2_x, Pixel* r2_y,
        Pixel* r2_xy)
{
    cudaFree(w_x);
    cudaFree(w_y);
    cudaFree(w_xy);

    cudaFree(w_bar_x);
    cudaFree(w_bar_y);
    cudaFree(w_bar_xy);

    cudaFree(w_previous_x);
    cudaFree(w_previous_y);
    cudaFree(w_previous_xy);

    cudaFree(r_x);
    cudaFree(r_y);
    cudaFree(r_xy);

    cudaFree(r2_x);
    cudaFree(r2_y);
    cudaFree(r2_xy);
}

template<typename Pixel>
void tgv_launch_gradient3_2d(
        Pixel* w_bar_x, Pixel* w_bar_y,
        Pixel* w_bar_xy,

        Pixel* r_x, Pixel* r_y,
        Pixel* r_xy, Pixel* q_temp,
        uint width, uint height,
        dim3 block_dimension,
        dim3 grid_dimension,
        dim3 grid_dimension_x,
        dim3 grid_dimension_y)
{
    forward_difference_x_2d<<<grid_dimension_x, block_dimension>>>(
          w_bar_x, r_x, width, height);

    forward_difference_y_2d<<<grid_dimension_y, block_dimension>>>(
          w_bar_y, r_y, width, height);

    forward_difference_x_2d<<<grid_dimension_x, block_dimension>>>(
          w_bar_xy, r_xy, width, height);
    forward_difference_y_2d<<<grid_dimension_y, block_dimension>>>(
          w_bar_xy, q_temp, width, height);
    addAndHalf_2d<<<grid_dimension, block_dimension>>>(
            r_xy, q_temp, r_xy,
            width, height);

    cudaCheckError( cudaDeviceSynchronize() );
}


template<typename Pixel>
void tgv_launch_divergence3_2d(
        Pixel* r_x, Pixel* r_y,
        Pixel* r_xy,

        Pixel* r2_x, Pixel* r2_y,
        Pixel* r2_xy,

        Pixel* r_temp,

        uint width, uint height,
        dim3 block_dimension,
        dim3 grid_dimension,
        dim3 grid_dimension_x,
        dim3 grid_dimension_y)
{
    backward_difference_x_2d<<<grid_dimension_x, block_dimension>>>(
      r_x, r2_x, width, height);

    backward_difference_y_2d<<<grid_dimension_y, block_dimension>>>(
      r_y, r2_y, width, height);

    backward_difference_x_2d<<<grid_dimension_x, block_dimension>>>(
      r_xy, r2_xy, width, height);
    backward_difference_y_2d<<<grid_dimension_y, block_dimension>>>(
      r_xy, r_temp, width, height);
    add_2d<<<grid_dimension, block_dimension>>>(r2_xy, r_temp, r2_xy, width, height);

    cudaCheckError( cudaDeviceSynchronize() );
}

template<typename Pixel>
__global__ void tgv3_kernel_part52_2d(
        Pixel* r2_x, Pixel* r2_y,
        Pixel* r2_xy,

        Pixel* r_x, Pixel* r_y,
        Pixel* r_xy,

        Pixel* w_x, Pixel* w_y,
        Pixel* w_xy,

        Pixel* w_previous_x, Pixel* w_previous_y,
        Pixel* w_previous_xy,

        const Pixel sigma, const Pixel alpha2,
        const uint width, const uint height) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height)
        return;

    r_x[index] = fmaf(sigma, r2_x[index], r_x[index]);
    r_y[index] = fmaf(sigma, r2_y[index], r_y[index]);
    r_xy[index] = fmaf(sigma, r2_xy[index], r_xy[index]);

    Pixel normalization =
            r_x[index] * r_x[index] +
            r_y[index] * r_y[index] +
            2 * r_xy[index] * r_xy[index];

    normalization = fmaxf(1, sqrtf(normalization) / alpha2);

    r_x[index] /= normalization;
    r_y[index] /= normalization;
    r_xy[index] /= normalization;

    w_previous_x[index] = w_x[index];
    w_previous_y[index] = w_y[index];
    w_previous_xy[index] = w_xy[index];
}

template<typename Pixel>
__global__ void tgv3_kernel_part62_2d(
        Pixel* r2_x, Pixel* r2_y,
        Pixel* r2_xy,

        Pixel* q_x, Pixel* q_y,
        Pixel* q_xy,

        Pixel* w_x, Pixel* w_y,
        Pixel* w_xy,

        Pixel* w_previous_x, Pixel* w_previous_y,
        Pixel* w_previous_xy,

        Pixel* w_bar_x, Pixel* w_bar_y,
        Pixel* w_bar_xy,

        const Pixel tau, const Pixel theta,
        const uint width, const uint height)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height)
        return;

    w_x[index] -= tau * (r2_x[index] - q_x[index]);
    w_y[index] -= tau * (r2_y[index] - q_y[index]);

    w_xy[index] -= tau * (r2_xy[index] - q_xy[index]);

    w_bar_x[index] = w_x[index] + theta*(w_x[index] - w_previous_x[index]);
    w_bar_y[index] = w_y[index] + theta*(w_y[index] - w_previous_y[index]);
    w_bar_xy[index] = w_xy[index] + theta*(w_xy[index] - w_previous_xy[index]);
}

#endif // TGV_3_COMMON_2D
