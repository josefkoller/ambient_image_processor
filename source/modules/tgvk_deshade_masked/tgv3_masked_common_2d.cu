#ifndef TGV_3_COMMON_MASKED_2D
#define TGV_3_COMMON_MASKED_2D

#include "tgv2_masked_common_2d.cu"

template<typename Pixel>
__global__ void tgv3_kernel_part5_masked_2d(
        Pixel* v_x, Pixel* v_y,
        Pixel* v_previous_x, Pixel* v_previous_y,
        Pixel* q_x,Pixel* q_y,
        Pixel* q_xy,
        Pixel* q_x2, Pixel* q_y2,
        Pixel* q_xy2,

        Pixel* w_x,Pixel* w_y,
        Pixel* w_xy,

        const Pixel sigma, const Pixel alpha0,

        Index* indices, IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    q_x2[pixel_index] = fmaf(sigma, q_x[pixel_index] - w_x[pixel_index], q_x2[pixel_index]);
    q_y2[pixel_index] = fmaf(sigma, q_y[pixel_index] - w_y[pixel_index], q_y2[pixel_index]);
    q_xy2[pixel_index] = fmaf(sigma,  q_xy[pixel_index] - w_xy[pixel_index], q_xy2[pixel_index]);

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
void tgv_launch_gradient3_masked_2d(
        Pixel* w_bar_x, Pixel* w_bar_y,
        Pixel* w_bar_xy,

        Pixel* r_x, Pixel* r_y,
        Pixel* r_xy, Pixel* q_temp,

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
    launch_forward_difference_x_masked(w_bar_x, r_x,
        block_dimension, right_grid_dimension, not_right_grid_dimension,
        right_indices, right_indices_count,
        not_right_indices, not_right_indices_count);

    launch_forward_difference_y_masked(w_bar_y, r_y, width,
        block_dimension, bottom_grid_dimension, not_bottom_grid_dimension,
        bottom_indices, bottom_indices_count,
        not_bottom_indices, not_bottom_indices_count);

    launch_forward_difference_x_masked(w_bar_xy, r_xy,
        block_dimension, right_grid_dimension, not_right_grid_dimension,
        right_indices, right_indices_count,
        not_right_indices, not_right_indices_count);
    launch_forward_difference_y_masked(w_bar_xy, q_temp, width,
        block_dimension, bottom_grid_dimension, not_bottom_grid_dimension,
        bottom_indices, bottom_indices_count,
        not_bottom_indices, not_bottom_indices_count);
    cudaCheckError( cudaDeviceSynchronize() );
    add_and_half_masked<<<masked_grid_dimension, block_dimension>>>(
            r_xy, q_temp,
            masked_indices, masked_indices_count);

    cudaCheckError( cudaDeviceSynchronize() );
}


template<typename Pixel>
void tgv_launch_divergence3_masked_2d(
        Pixel* r_x, Pixel* r_y,
        Pixel* r_xy,

        Pixel* r2_x, Pixel* r2_y,
        Pixel* r2_xy,

        Pixel* r_temp,

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
    launch_backward_difference_x_masked(r_x, r2_x,
        block_dimension, left_grid_dimension, not_left_grid_dimension,
        left_indices, left_indices_count,
        not_left_indices, not_left_indices_count);

    launch_backward_difference_y_masked(r_y, r2_y, width,
        block_dimension, top_grid_dimension, not_top_grid_dimension,
        top_indices, top_indices_count,
        not_top_indices, not_top_indices_count);

    launch_backward_difference_x_masked(r_xy, r2_xy,
        block_dimension, left_grid_dimension, not_left_grid_dimension,
        left_indices, left_indices_count,
        not_left_indices, not_left_indices_count);
    launch_backward_difference_y_masked(r_xy, r_temp, width,
        block_dimension, top_grid_dimension, not_top_grid_dimension,
        top_indices, top_indices_count,
        not_top_indices, not_top_indices_count);
    cudaCheckError( cudaDeviceSynchronize() );
    add_masked<<<masked_grid_dimension, block_dimension>>>(
        r2_xy, r_temp, masked_indices, masked_indices_count);

    cudaCheckError( cudaDeviceSynchronize() );
}

template<typename Pixel>
__global__ void tgv3_kernel_part52_masked_2d(
        Pixel* r2_x, Pixel* r2_y,
        Pixel* r2_xy,

        Pixel* r_x, Pixel* r_y,
        Pixel* r_xy,

        Pixel* w_x, Pixel* w_y,
        Pixel* w_xy,

        Pixel* w_previous_x, Pixel* w_previous_y,
        Pixel* w_previous_xy,

        const Pixel sigma, const Pixel alpha2,

        Index* indices,  IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    r_x[pixel_index] = fmaf(sigma, r2_x[pixel_index], r_x[pixel_index]);
    r_y[pixel_index] = fmaf(sigma, r2_y[pixel_index], r_y[pixel_index]);
    r_xy[pixel_index] = fmaf(sigma, r2_xy[pixel_index], r_xy[pixel_index]);

    Pixel normalization =
            r_x[pixel_index] * r_x[pixel_index] +
            r_y[pixel_index] * r_y[pixel_index] +
            2 * r_xy[pixel_index] * r_xy[pixel_index];

    normalization = fmaxf(1, sqrtf(normalization) / alpha2);

    r_x[pixel_index] /= normalization;
    r_y[pixel_index] /= normalization;
    r_xy[pixel_index] /= normalization;

    w_previous_x[pixel_index] = w_x[pixel_index];
    w_previous_y[pixel_index] = w_y[pixel_index];
    w_previous_xy[pixel_index] = w_xy[pixel_index];
}

template<typename Pixel>
__global__ void tgv3_kernel_part62_masked_2d(
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

        Index* indices,  IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    w_x[pixel_index] -= tau * (r2_x[pixel_index] - q_x[pixel_index]);
    w_y[pixel_index] -= tau * (r2_y[pixel_index] - q_y[pixel_index]);
    w_xy[pixel_index] -= tau * (r2_xy[pixel_index] - q_xy[pixel_index]);

    w_bar_x[pixel_index] = w_x[pixel_index] + theta*(w_x[pixel_index] - w_previous_x[pixel_index]);
    w_bar_y[pixel_index] = w_y[pixel_index] + theta*(w_y[pixel_index] - w_previous_y[pixel_index]);
    w_bar_xy[pixel_index] = w_xy[pixel_index] + theta*(w_xy[pixel_index] - w_previous_xy[pixel_index]);
}

#endif // TGV_3_COMMON_MASKED_2D
