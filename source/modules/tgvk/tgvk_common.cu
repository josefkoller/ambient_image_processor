#ifndef TGV_K_COMMON
#define TGV_K_COMMON

#include "tgv2_common.cu"

template<typename Pixel>
__global__ void tgvk_kernel_part5(
        Pixel* r2_x, Pixel* r2_y, Pixel* r2_z,
        Pixel* r2_xy, Pixel* r2_xz, Pixel* r2_yz,

        Pixel* r_x, Pixel* r_y, Pixel* r_z,
        Pixel* r_xy, Pixel* r_xz, Pixel* r_yz,

        Pixel* w_x, Pixel* w_y, Pixel* w_z,
        Pixel* w_xy, Pixel* w_xz, Pixel* w_yz,

        Pixel* w_previous_x, Pixel* w_previous_y, Pixel* w_previous_z,
        Pixel* w_previous_xy, Pixel* w_previous_xz, Pixel* w_previous_yz,

        Pixel* w_prime_x, Pixel* w_prime_y, Pixel* w_prime_z,
        Pixel* w_prime_xy, Pixel* w_prime_xz, Pixel* w_prime_yz,

        const Pixel sigma, const Pixel alpha2,
        const uint width, const uint height, const uint depth) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    r_x[index] += sigma * (r2_x[index] - w_prime_x[index]);
    r_y[index] += sigma * (r2_y[index] - w_prime_y[index]);
    r_xy[index] += sigma * (r2_xy[index] - w_prime_xy[index]);

    if(depth > 1) {
        r_z[index] += sigma * (r2_z[index] - w_prime_z[index]);
        r_xz[index] += sigma * (r2_xz[index] - w_prime_xz[index]);
        r_yz[index] += sigma * (r2_yz[index] - w_prime_yz[index]);
    }

    Pixel normalization =
            r_x[index] * r_x[index] +
            r_y[index] * r_y[index] +
            2 * r_xy[index] * r_xy[index];

    if(depth > 1)
        normalization += r_z[index] * r_z[index] +
                2 * r_xz[index] * r_xz[index] +
                2 * r_yz[index] * r_yz[index];

    normalization = fmax(1, sqrt(normalization) / alpha2);

    r_x[index] /= normalization;
    r_y[index] /= normalization;
    r_xy[index] /= normalization;
    if(depth > 1) {
        r_z[index] /= normalization;
        r_xz[index] /= normalization;
        r_yz[index] /= normalization;

        w_previous_z[index] = w_z[index];
        w_previous_xz[index] = w_xz[index];
        w_previous_yz[index] = w_yz[index];
    }

    w_previous_x[index] = w_x[index];
    w_previous_y[index] = w_y[index];
    w_previous_xy[index] = w_xy[index];
}


template<typename Pixel>
void tgv_launch_gradient3_backward(
        Pixel* w_bar_x, Pixel* w_bar_y, Pixel* w_bar_z,
        Pixel* w_bar_xy, Pixel* w_bar_xz, Pixel* w_bar_yz,

        Pixel* r_x, Pixel* r_y, Pixel* r_z,
        Pixel* r_xy, Pixel* r_xz, Pixel* r_yz,  Pixel* q_temp,
        uint width, uint height, uint depth,
        dim3 block_dimension,
        dim3 grid_dimension,
        dim3 grid_dimension_x,
        dim3 grid_dimension_y,
        dim3 grid_dimension_z)
{
    backward_difference_x<<<grid_dimension_x, block_dimension>>>(
          w_bar_x, r_x, width, height, depth);

    backward_difference_y<<<grid_dimension_y, block_dimension>>>(
          w_bar_y, r_y, width, height, depth);

    backward_difference_x<<<grid_dimension_x, block_dimension>>>(
          w_bar_xy, r_xy, width, height, depth);
    backward_difference_y<<<grid_dimension_y, block_dimension>>>(
          w_bar_xy, q_temp, width, height, depth);
    addAndHalf<<<grid_dimension, block_dimension>>>(
            r_xy, q_temp, r_xy,
            width, height, depth);

    if(depth > 1) {
        backward_difference_z<<<grid_dimension_z, block_dimension>>>(
              w_bar_z, r_z, width, height, depth);

        backward_difference_x<<<grid_dimension_x, block_dimension>>>(
              w_bar_xz, r_xz, width, height, depth);
        backward_difference_z<<<grid_dimension_z, block_dimension>>>(
              w_bar_xz, q_temp, width, height, depth);
        addAndHalf<<<grid_dimension, block_dimension>>>(
                r_xz, q_temp, r_xz,
                width, height, depth);

        backward_difference_y<<<grid_dimension_y, block_dimension>>>(
              w_bar_yz, r_yz, width, height, depth);
        backward_difference_z<<<grid_dimension_z, block_dimension>>>(
              w_bar_yz, q_temp, width, height, depth);
        addAndHalf<<<grid_dimension, block_dimension>>>(
                r_yz, q_temp, r_yz,
                width, height, depth);
    }
    cudaCheckError( cudaDeviceSynchronize() );
}


template<typename Pixel>
void tgv_launch_divergence3_forward(
        Pixel* r_x, Pixel* r_y, Pixel* r_z,
        Pixel* r_xy, Pixel* r_xz, Pixel* r_yz,

        Pixel* r2_x, Pixel* r2_y, Pixel* r2_z,
        Pixel* r2_xy, Pixel* r2_xz, Pixel* r2_yz,

        Pixel* r_temp,

        uint width, uint height, uint depth,
        dim3 block_dimension,
        dim3 grid_dimension,
        dim3 grid_dimension_x,
        dim3 grid_dimension_y,
        dim3 grid_dimension_z)
{
    forward_difference_x<<<grid_dimension_x, block_dimension>>>(
      r_x, r2_x, width, height, depth);

    forward_difference_y<<<grid_dimension_y, block_dimension>>>(
      r_y, r2_y, width, height, depth);

    forward_difference_x<<<grid_dimension_x, block_dimension>>>(
      r_xy, r2_xy, width, height, depth);
    forward_difference_y<<<grid_dimension_y, block_dimension>>>(
      r_xy, r_temp, width, height, depth);
    add<<<grid_dimension, block_dimension>>>(r2_xy, r_temp, r2_xy, width, height, depth);

    if(depth > 1) {
        forward_difference_z<<<grid_dimension_z, block_dimension>>>(
          r_z, r2_z, width, height, depth);

        forward_difference_x<<<grid_dimension_x, block_dimension>>>(
          r_xz, r2_xz, width, height, depth);
        forward_difference_z<<<grid_dimension_z, block_dimension>>>(
          r_xz, r_temp, width, height, depth);
        add<<<grid_dimension, block_dimension>>>(r2_xz, r_temp, r2_xz, width, height, depth);

        forward_difference_y<<<grid_dimension_y, block_dimension>>>(
          r_yz, r2_yz, width, height, depth);
        forward_difference_z<<<grid_dimension_z, block_dimension>>>(
          r_yz, r_temp, width, height, depth);
        add<<<grid_dimension, block_dimension>>>(r2_yz, r_temp, r2_yz, width, height, depth);
    }
    cudaCheckError( cudaDeviceSynchronize() );
}

#endif // TGV_K_COMMON
