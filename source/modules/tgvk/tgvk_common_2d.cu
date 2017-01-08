/*
    Ambient Image Processor - A tool to perform several imaging tasks
    
    Copyright (C) 2016 Josef Koller

    https://github.com/josefkoller/ambient_image_processor    
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef TGV_K_COMMON_2D
#define TGV_K_COMMON_2D

#include "tgv3_common_2d.cu"

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
