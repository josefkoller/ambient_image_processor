#ifndef TGV_K_COMMON_2D
#define TGV_K_COMMON_2D

#include "tgv2_common_2d.cu"

template<typename Pixel>
__global__ void tgvk_kernel_part5_2d(
        Pixel* r2_x, Pixel* r2_y,
        Pixel* r2_xy,

        Pixel* r_x, Pixel* r_y,
        Pixel* r_xy,

        Pixel* w_x, Pixel* w_y,
        Pixel* w_xy,

        Pixel* w_previous_x, Pixel* w_previous_y,
        Pixel* w_previous_xy,

        Pixel* w_prime_x, Pixel* w_prime_y,
        Pixel* w_prime_xy,

        const Pixel sigma, const Pixel alpha2,
        const uint width, const uint height) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height)
        return;

    r_x[index] = fmaf(sigma, r2_x[index] - w_prime_x[index], r_x[index]);
    r_y[index] = fmaf(sigma, r2_y[index] - w_prime_y[index], r_y[index]);
    r_xy[index] = fmaf(sigma, r2_xy[index] - w_prime_xy[index], r_xy[index]);

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
void tgv_launch_gradient3_backward_2d(
        Pixel* w_bar_x, Pixel* w_bar_y,
        Pixel* w_bar_xy,

        Pixel* r_x, Pixel* r_y,
        Pixel* r_xy,  Pixel* q_temp,
        uint width, uint height,
        dim3 block_dimension,
        dim3 grid_dimension,
        dim3 grid_dimension_x,
        dim3 grid_dimension_y)
{
    backward_difference_x_2d<<<grid_dimension_x, block_dimension>>>(
          w_bar_x, r_x, width, height);

    backward_difference_y_2d<<<grid_dimension_y, block_dimension>>>(
          w_bar_y, r_y, width, height);

    backward_difference_x_2d<<<grid_dimension_x, block_dimension>>>(
          w_bar_xy, r_xy, width, height);
    backward_difference_y_2d<<<grid_dimension_y, block_dimension>>>(
          w_bar_xy, q_temp, width, height);
    addAndHalf_2d<<<grid_dimension, block_dimension>>>(
            r_xy, q_temp, r_xy,
            width, height);

    cudaCheckError( cudaDeviceSynchronize() );
}


template<typename Pixel>
void tgv_launch_divergence3_forward_2d(
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
    forward_difference_x_2d<<<grid_dimension_x, block_dimension>>>(
      r_x, r2_x, width, height);

    forward_difference_y_2d<<<grid_dimension_y, block_dimension>>>(
      r_y, r2_y, width, height);

    forward_difference_x_2d<<<grid_dimension_x, block_dimension>>>(
      r_xy, r2_xy, width, height);
    forward_difference_y_2d<<<grid_dimension_y, block_dimension>>>(
      r_xy, r_temp, width, height);
    add_2d<<<grid_dimension, block_dimension>>>(r2_xy, r_temp, r2_xy, width, height);

    cudaCheckError( cudaDeviceSynchronize() );
}

#endif // TGV_K_COMMON_2D
