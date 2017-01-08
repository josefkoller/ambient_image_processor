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

#ifndef TGV_K_MASKED_COMMON
#define TGV_K_MASKED_COMMON

#include "tgv3_masked_common.cu"

template<typename Pixel>
__global__ void tgvk_kernel_part5_masked(
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

        Index* indices, IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    r_x[pixel_index] = fmaf(sigma, r2_x[pixel_index] - w_prime_x[pixel_index], r_x[pixel_index]);
    r_y[pixel_index] = fmaf(sigma, r2_y[pixel_index] - w_prime_y[pixel_index], r_y[pixel_index]);
    r_xy[pixel_index] = fmaf(sigma, r2_xy[pixel_index] - w_prime_xy[pixel_index], r_xy[pixel_index]);

    r_z[pixel_index] = fmaf(sigma, r2_z[pixel_index] - w_prime_z[pixel_index], r_z[pixel_index]);
    r_xz[pixel_index] = fmaf(sigma, r2_xz[pixel_index] - w_prime_xz[pixel_index], r_xz[pixel_index]);
    r_yz[pixel_index] = fmaf(sigma, r2_yz[pixel_index] - w_prime_yz[pixel_index], r_yz[pixel_index]);

    Pixel normalization =
            r_x[pixel_index] * r_x[pixel_index] +
            r_y[pixel_index] * r_y[pixel_index] +
            2 * r_xy[pixel_index] * r_xy[pixel_index] +
            r_z[pixel_index] * r_z[pixel_index] +
            2 * r_xz[pixel_index] * r_xz[pixel_index] +
            2 * r_yz[pixel_index] * r_yz[pixel_index];

    normalization = fmaxf(1, sqrtf(normalization) / alpha2);

    r_x[pixel_index] /= normalization;
    r_y[pixel_index] /= normalization;
    r_xy[pixel_index] /= normalization;

    r_z[pixel_index] /= normalization;
    r_xz[pixel_index] /= normalization;
    r_yz[pixel_index] /= normalization;

    w_previous_z[pixel_index] = w_z[pixel_index];
    w_previous_xz[pixel_index] = w_xz[pixel_index];
    w_previous_yz[pixel_index] = w_yz[pixel_index];

    w_previous_x[pixel_index] = w_x[pixel_index];
    w_previous_y[pixel_index] = w_y[pixel_index];
    w_previous_xy[pixel_index] = w_xy[pixel_index];
}


template<typename Pixel>
void tgv_launch_gradient3_backward_masked(
        Pixel* w_bar_x, Pixel* w_bar_y, Pixel* w_bar_z,
        Pixel* w_bar_xy, Pixel* w_bar_xz, Pixel* w_bar_yz,

        Pixel* r_x, Pixel* r_y, Pixel* r_z,
        Pixel* r_xy, Pixel* r_xz, Pixel* r_yz,  Pixel* q_temp,

        Size width, Size width_x_height,
        GridDimension block_dimension,
        GridDimension masked_grid_dimension,
        GridDimension left_grid_dimension,
        GridDimension not_left_grid_dimension,
        GridDimension top_grid_dimension,
        GridDimension not_top_grid_dimension,
        GridDimension front_grid_dimension,
        GridDimension not_front_grid_dimension,
        Index* masked_indices, IndexCount masked_indices_count,
        Index* left_indices, IndexCount left_indices_count,
        Index* not_left_indices, IndexCount not_left_indices_count,
        Index* top_indices, IndexCount top_indices_count,
        Index* not_top_indices, IndexCount not_top_indices_count,
        Index* front_indices, IndexCount front_indices_count,
        Index* not_front_indices, IndexCount not_front_indices_count)
{
    launch_backward_difference_x_masked(w_bar_x, r_x,
        block_dimension, left_grid_dimension, not_left_grid_dimension,
        left_indices, left_indices_count,
        not_left_indices, not_left_indices_count);

    launch_backward_difference_y_masked(w_bar_y, r_y, width,
        block_dimension, top_grid_dimension, not_top_grid_dimension,
        top_indices, top_indices_count,
        not_top_indices, not_top_indices_count);

    launch_backward_difference_x_masked(w_bar_xy, r_xy,
        block_dimension, left_grid_dimension, not_left_grid_dimension,
        left_indices, left_indices_count,
        not_left_indices, not_left_indices_count);
    launch_backward_difference_y_masked(w_bar_xy, q_temp, width,
        block_dimension, top_grid_dimension, not_top_grid_dimension,
        top_indices, top_indices_count,
        not_top_indices, not_top_indices_count);
    cudaCheckError( cudaDeviceSynchronize() );
    add_and_half_masked<<<masked_grid_dimension, block_dimension>>>(
            r_xy, q_temp, masked_indices, masked_indices_count);

    launch_backward_difference_z_masked(w_bar_z, r_z, width_x_height,
        block_dimension, front_grid_dimension, not_front_grid_dimension,
        front_indices, front_indices_count,
        not_front_indices, not_front_indices_count);

    launch_backward_difference_x_masked(w_bar_xz, r_xz,
        block_dimension, left_grid_dimension, not_left_grid_dimension,
        left_indices, left_indices_count,
        not_left_indices, not_left_indices_count);
    launch_backward_difference_z_masked(w_bar_xz, q_temp, width_x_height,
        block_dimension, front_grid_dimension, not_front_grid_dimension,
        front_indices, front_indices_count,
        not_front_indices, not_front_indices_count);
    cudaCheckError( cudaDeviceSynchronize() );
    add_and_half_masked<<<masked_grid_dimension, block_dimension>>>(
            r_xz, q_temp, masked_indices, masked_indices_count);

    launch_backward_difference_y_masked(w_bar_yz, r_yz, width,
        block_dimension, top_grid_dimension, not_top_grid_dimension,
        top_indices, top_indices_count,
        not_top_indices, not_top_indices_count);
    launch_backward_difference_z_masked(w_bar_yz, q_temp, width_x_height,
        block_dimension, front_grid_dimension, not_front_grid_dimension,
        front_indices, front_indices_count,
        not_front_indices, not_front_indices_count);
    cudaCheckError( cudaDeviceSynchronize() );
    add_and_half_masked<<<masked_grid_dimension, block_dimension>>>(
            r_yz, q_temp, masked_indices, masked_indices_count);

    cudaCheckError( cudaDeviceSynchronize() );
}


template<typename Pixel>
void tgv_launch_divergence3_forward_masked(
        Pixel* r_x, Pixel* r_y, Pixel* r_z,
        Pixel* r_xy, Pixel* r_xz, Pixel* r_yz,

        Pixel* r2_x, Pixel* r2_y, Pixel* r2_z,
        Pixel* r2_xy, Pixel* r2_xz, Pixel* r2_yz,

        Pixel* r_temp,

        Size width, Size width_x_height,

        GridDimension block_dimension,
        GridDimension right_grid_dimension,
        GridDimension not_right_grid_dimension,
        GridDimension bottom_grid_dimension,
        GridDimension not_bottom_grid_dimension,
        GridDimension back_grid_dimension,
        GridDimension not_back_grid_dimension,
        GridDimension masked_grid_dimension,

        Index* right_indices, IndexCount right_indices_count,
        Index* not_right_indices, IndexCount not_right_indices_count,
        Index* bottom_indices, IndexCount bottom_indices_count,
        Index* not_bottom_indices, IndexCount not_bottom_indices_count,
        Index* back_indices, IndexCount back_indices_count,
        Index* not_back_indices, IndexCount not_back_indices_count,
        Index* masked_indices, IndexCount masked_indices_count)
{
    launch_forward_difference_x_masked(r_x, r2_x,
        block_dimension, right_grid_dimension, not_right_grid_dimension,
        right_indices, right_indices_count,
        not_right_indices, not_right_indices_count);

    launch_forward_difference_y_masked(r_y, r2_y, width,
        block_dimension, bottom_grid_dimension, not_bottom_grid_dimension,
        bottom_indices, bottom_indices_count,
        not_bottom_indices, not_bottom_indices_count);

    launch_forward_difference_x_masked(r_xy, r2_xy,
        block_dimension, right_grid_dimension, not_right_grid_dimension,
        right_indices, right_indices_count,
        not_right_indices, not_right_indices_count);
    launch_forward_difference_y_masked(r_xy, r_temp, width,
        block_dimension, bottom_grid_dimension, not_bottom_grid_dimension,
        bottom_indices, bottom_indices_count,
        not_bottom_indices, not_bottom_indices_count);
    cudaCheckError( cudaDeviceSynchronize() );
    add_masked<<<masked_grid_dimension, block_dimension>>>(
        r2_xy, r_temp, masked_indices, masked_indices_count);

    launch_forward_difference_z_masked(r_z, r2_z, width_x_height,
        block_dimension, back_grid_dimension, not_back_grid_dimension,
        back_indices, back_indices_count,
        not_back_indices, not_back_indices_count);

    launch_forward_difference_x_masked(r_xz, r2_xz,
        block_dimension, right_grid_dimension, not_right_grid_dimension,
        right_indices, right_indices_count,
        not_right_indices, not_right_indices_count);
    launch_forward_difference_z_masked(r_xz, r_temp, width_x_height,
        block_dimension, back_grid_dimension, not_back_grid_dimension,
        back_indices, back_indices_count,
        not_back_indices, not_back_indices_count);
    cudaCheckError( cudaDeviceSynchronize() );
    add_masked<<<masked_grid_dimension, block_dimension>>>(
        r2_xz, r_temp, masked_indices, masked_indices_count);

    launch_forward_difference_y_masked(r_yz, r2_yz, width,
        block_dimension, bottom_grid_dimension, not_bottom_grid_dimension,
        bottom_indices, bottom_indices_count,
        not_bottom_indices, not_bottom_indices_count);
    launch_forward_difference_z_masked(r_yz, r_temp, width_x_height,
        block_dimension, back_grid_dimension, not_back_grid_dimension,
        back_indices, back_indices_count,
        not_back_indices, not_back_indices_count);
    cudaCheckError( cudaDeviceSynchronize() );
    add_masked<<<masked_grid_dimension, block_dimension>>>(
        r2_yz, r_temp, masked_indices, masked_indices_count);

    cudaCheckError( cudaDeviceSynchronize() );
}

#endif // TGV_K_MASKED_COMMON
