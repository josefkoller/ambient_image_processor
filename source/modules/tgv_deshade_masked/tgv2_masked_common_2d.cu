#ifndef TGV_2_MASKED_COMMON_2D
#define TGV_2_MASKED_COMMON_2D

#include "cuda_helper.cuh"
#include "tgv_masked_common_2d.cu"
#include "tgv2_masked_common.cu"

template<typename Pixel>
__global__ void tgv_kernel_part22_masked_2d(
        Pixel* v_bar_x, Pixel* v_bar_y,
        Pixel* p_x, Pixel* p_y,
        Pixel* p_xx, Pixel* p_yy,
        const Pixel sigma, const Pixel alpha1, Pixel* u_previous, Pixel* u,

        Index* indices, IndexCount indices_count) {

    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    p_xx[pixel_index] = fmaf(sigma, p_x[pixel_index] - v_bar_x[pixel_index], p_xx[pixel_index]);
    p_yy[pixel_index] = fmaf(sigma, p_y[pixel_index] - v_bar_y[pixel_index], p_yy[pixel_index]);

    Pixel normalization = sqrtf(p_xx[pixel_index] * p_xx[pixel_index] + p_yy[pixel_index] * p_yy[pixel_index]);

    normalization = fmaxf(1, normalization/alpha1);

    p_xx[pixel_index] /= normalization;
    p_yy[pixel_index] /= normalization;

    u_previous[pixel_index] = u[pixel_index];
}

template<typename Pixel>
void tgv_launch_gradient2_masked_2d(
        Pixel* v_bar_x, Pixel* v_bar_y,
        Pixel* q_x, Pixel* q_y,
        Pixel* q_xy, Pixel* q_temp,
        Size width,
        GridDimension block_dimension,
        GridDimension masked_grid_dimension,
        GridDimension left_grid_dimension,
        GridDimension not_left_grid_dimension,
        GridDimension top_grid_dimension,
        GridDimension not_top_grid_dimension,
        Index* masked_indices, IndexCount masked_indices_count,
        Index* left_indices, IndexCount left_indices_count,
        Index* not_left_indices, IndexCount not_left_indices_count,
        Index* top_indices, IndexCount top_indices_count,
        Index* not_top_indices, IndexCount not_top_indices_count)
{
    launch_backward_difference_x_masked(v_bar_x, q_x,
        block_dimension, left_grid_dimension, not_left_grid_dimension,
        left_indices, left_indices_count,
        not_left_indices, not_left_indices_count);

    launch_backward_difference_y_masked(v_bar_y, q_y, width,
        block_dimension, top_grid_dimension, not_top_grid_dimension,
        top_indices, top_indices_count,
        not_top_indices, not_top_indices_count);

    launch_backward_difference_x_masked(v_bar_y, q_xy,
        block_dimension, left_grid_dimension, not_left_grid_dimension,
        left_indices, left_indices_count,
        not_left_indices, not_left_indices_count);
    launch_backward_difference_y_masked(v_bar_x, q_temp, width,
        block_dimension, top_grid_dimension, not_top_grid_dimension,
        top_indices, top_indices_count,
        not_top_indices, not_top_indices_count);
    cudaCheckError( cudaDeviceSynchronize() );
    add_and_half_masked<<<masked_grid_dimension, block_dimension>>>(
            q_xy, q_temp,
            masked_indices, masked_indices_count);

    cudaCheckError( cudaDeviceSynchronize() );
}

template<typename Pixel>
__global__ void tgv_kernel_part5_masked_2d(
        Pixel* v_x,Pixel* v_y,
        Pixel* v_previous_x, Pixel* v_previous_y,
        Pixel* q_x,Pixel* q_y,
        Pixel* q_xy,
        Pixel* q_x2, Pixel* q_y2,
        Pixel* q_xy2,
        const Pixel sigma, const Pixel alpha0,
        Index* indices,  IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    q_x2[pixel_index] = fmaf(sigma, q_x[pixel_index], q_x2[pixel_index]);
    q_y2[pixel_index] = fmaf(sigma, q_y[pixel_index], q_y2[pixel_index]);
    q_xy2[pixel_index] = fmaf(sigma, q_xy[pixel_index], q_xy2[pixel_index]);

    Pixel normalization =
            q_x2[pixel_index] * q_x2[pixel_index] +
            q_y2[pixel_index] * q_y2[pixel_index] +
            2 * q_xy2[pixel_index] * q_xy2[pixel_index];

    normalization = fmaxf(1, sqrtf(normalization) / alpha0);

    q_x2[pixel_index] /= normalization;
    q_y2[pixel_index] /= normalization;
    q_xy2[pixel_index] /= normalization;

    v_previous_x[pixel_index] = v_x[pixel_index];
    v_previous_y[pixel_index] = v_y[pixel_index];
}

template<typename Pixel>
void tgv_launch_divergence2_masked_2d(
        Pixel* q_x, Pixel* q_y,
        Pixel* q_xy,
        Pixel* q_x2, Pixel* q_y2,
        Pixel* q_temp,
        Size width,
        GridDimension block_dimension,
        GridDimension right_grid_dimension,
        GridDimension not_right_grid_dimension,
        GridDimension bottom_grid_dimension,
        GridDimension not_bottom_grid_dimension,
        GridDimension masked_grid_dimension,
        Index* right_indices, IndexCount right_indices_count,
        Index* not_right_indices, IndexCount not_right_indices_count,
        Index* bottom_indices, IndexCount bottom_indices_count,
        Index* not_bottom_indices, IndexCount not_bottom_indices_count,
        Index* masked_indices, IndexCount masked_indices_count)
{
    launch_forward_difference_x_masked(q_x, q_x2,
        block_dimension, right_grid_dimension, not_right_grid_dimension,
        right_indices, right_indices_count,
        not_right_indices, not_right_indices_count);
    launch_forward_difference_y_masked(q_xy, q_temp, width,
        block_dimension, bottom_grid_dimension, not_bottom_grid_dimension,
        bottom_indices, bottom_indices_count,
        not_bottom_indices, not_bottom_indices_count);
    cudaCheckError( cudaDeviceSynchronize() );
    add_masked<<<masked_grid_dimension, block_dimension>>>(
        q_x2, q_temp, masked_indices, masked_indices_count);

    launch_forward_difference_y_masked(q_y, q_y2, width,
        block_dimension, bottom_grid_dimension, not_bottom_grid_dimension,
        bottom_indices, bottom_indices_count,
        not_bottom_indices, not_bottom_indices_count);
    launch_forward_difference_x_masked(q_xy, q_temp,
        block_dimension, right_grid_dimension, not_right_grid_dimension,
        right_indices, right_indices_count,
        not_right_indices, not_right_indices_count);
    cudaCheckError( cudaDeviceSynchronize() );
    add_masked<<<masked_grid_dimension, block_dimension>>>(
        q_y2, q_temp, masked_indices, masked_indices_count);

    cudaCheckError( cudaDeviceSynchronize() );
}


template<typename Pixel>
__global__ void tgv_kernel_part6_masked_2d(
        Pixel*  v_x, Pixel*  v_y,
        Pixel* q_x2, Pixel* q_y2,
        Pixel* p_x, Pixel* p_y,
        Pixel* v_previous_x, Pixel* v_previous_y,
        Pixel* v_bar_x, Pixel* v_bar_y,
        const Pixel tau, const Pixel theta,
        Index* indices,  IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    v_x[pixel_index] -= tau * (q_x2[pixel_index] - p_x[pixel_index]);
    v_y[pixel_index] -= tau * (q_y2[pixel_index] - p_y[pixel_index]);

    v_bar_x[pixel_index] = v_x[pixel_index] + theta*(v_x[pixel_index] - v_previous_x[pixel_index]);
    v_bar_y[pixel_index] = v_y[pixel_index] + theta*(v_y[pixel_index] - v_previous_y[pixel_index]);
}

template<typename Pixel>
__global__ void tgv_kernel_part4_tgv2_l1_masked_2d(
        Pixel* p_x, Pixel* p_y,
        Pixel* u, Pixel* f,
        const Pixel tau_x_lambda, const Pixel tau,
        Pixel* u_previous, const Pixel theta, Pixel* u_bar,
        Index* indices,  IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    u[pixel_index] -= tau * (p_x[pixel_index] + p_y[pixel_index]);

    const Pixel u_minus_f = u[pixel_index] - f[pixel_index];
    if(u_minus_f > tau_x_lambda)
        u[pixel_index] -= tau_x_lambda;
    else if(u_minus_f < -tau_x_lambda)
        u[pixel_index] += tau_x_lambda;
    else
        u[pixel_index] = f[pixel_index];

    u_bar[pixel_index] = u[pixel_index] + theta*(u[pixel_index] - u_previous[pixel_index]);
}



#endif // TGV_2_MASKED_COMMON_2D
