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

#ifndef TGV_2_MASKED_COMMON
#define TGV_2_MASKED_COMMON

#include "cuda_helper.cuh"
#include "tgv_masked_common.cu"

template<typename Pixel>
__global__ void add_and_half_masked(
        Pixel* result, Pixel* second,
        Index* indices, IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    result[pixel_index] = (result[pixel_index] + second[pixel_index]) * 0.5;
}

template<typename Pixel>
__global__ void add_masked(
        Pixel* result, Pixel* second,
        Index* indices, IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    result[pixel_index] += second[pixel_index];
}

template<typename Pixel>
__global__ void tgv_kernel_part22_masked(
        Pixel* v_bar_x, Pixel* v_bar_y, Pixel* v_bar_z,
        Pixel* p_x, Pixel* p_y, Pixel* p_z,
        Pixel* p_xx, Pixel* p_yy, Pixel* p_zz,
        const Pixel sigma, const Pixel alpha1, Pixel* u_previous, Pixel* u,

        Index* indices, IndexCount indices_count) {

    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    /*
    * Matlab Code:
    p = p + sigma*(nabla*u_bar - v_bar);
    norm_p  = sqrt(p(1:N).^2 + p(N+1:2*N).^2 +  p(2*N+1:3*N).^2);
    p = p./max(1,[norm_p; norm_p; norm_p]/alpha1);

    u_old = u;
    */

    p_xx[pixel_index] = fmaf(sigma, p_x[pixel_index] - v_bar_x[pixel_index], p_xx[pixel_index]);
    p_yy[pixel_index] = fmaf(sigma, p_y[pixel_index] - v_bar_y[pixel_index], p_yy[pixel_index]);
    p_zz[pixel_index] = fmaf(sigma, p_z[pixel_index] - v_bar_z[pixel_index], p_zz[pixel_index]);

    Pixel normalization = norm3df(p_xx[pixel_index], p_yy[pixel_index], p_zz[pixel_index]);

    normalization = fmaxf(1, normalization/alpha1);

    p_xx[pixel_index] /= normalization;
    p_yy[pixel_index] /= normalization;
    p_zz[pixel_index] /= normalization;

    u_previous[pixel_index] = u[pixel_index];
}

template<typename Pixel>
void tgv_launch_gradient2_masked(
        Pixel* v_bar_x, Pixel* v_bar_y, Pixel* v_bar_z,
        Pixel* q_x, Pixel* q_y, Pixel* q_z,
        Pixel* q_xy, Pixel* q_xz, Pixel* q_yz,  Pixel* q_temp,
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

    launch_backward_difference_z_masked(v_bar_z, q_z, width_x_height,
        block_dimension, front_grid_dimension, not_front_grid_dimension,
        front_indices, front_indices_count,
        not_front_indices, not_front_indices_count);

    launch_backward_difference_x_masked(v_bar_z, q_xz,
        block_dimension, left_grid_dimension, not_left_grid_dimension,
        left_indices, left_indices_count,
        not_left_indices, not_left_indices_count);
    launch_backward_difference_z_masked(v_bar_x, q_temp, width_x_height,
        block_dimension, front_grid_dimension, not_front_grid_dimension,
        front_indices, front_indices_count,
        not_front_indices, not_front_indices_count);
    cudaCheckError( cudaDeviceSynchronize() );
    add_and_half_masked<<<masked_grid_dimension, block_dimension>>>(
            q_xz, q_temp,
            masked_indices, masked_indices_count);

    launch_backward_difference_y_masked(v_bar_z, q_yz, width,
        block_dimension, top_grid_dimension, not_top_grid_dimension,
        top_indices, top_indices_count,
        not_top_indices, not_top_indices_count);
    launch_backward_difference_z_masked(v_bar_y, q_temp, width_x_height,
        block_dimension, front_grid_dimension, not_front_grid_dimension,
        front_indices, front_indices_count,
        not_front_indices, not_front_indices_count);
    cudaCheckError( cudaDeviceSynchronize() );
    add_and_half_masked<<<masked_grid_dimension, block_dimension>>>(
            q_yz, q_temp,
            masked_indices, masked_indices_count);

    cudaCheckError( cudaDeviceSynchronize() );
}

template<typename Pixel>
__global__ void tgv_kernel_part5_masked(
        Pixel* v_x,Pixel* v_y,Pixel* v_z,
        Pixel* v_previous_x, Pixel* v_previous_y, Pixel* v_previous_z,
        Pixel* q_x,Pixel* q_y,Pixel* q_z,
        Pixel* q_xy,Pixel* q_xz,Pixel* q_yz,
        Pixel* q_x2, Pixel* q_y2, Pixel* q_z2,
        Pixel* q_xy2, Pixel* q_xz2, Pixel* q_yz2,
        const Pixel sigma, const Pixel alpha0,
        Index* indices,  IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

       /*
        * Matlab Code:
        % dual update q
        q = q + sigma*nabla_second*v_bar;
        norm_q = sqrt(q(1:N).^2 + q(N+1:2*N).^2 + q(2*N+1:3*N).^2 + ... % main diagonal
            2*q(3*N+1:4*N).^2 + 2*q(4*N+1:5*N).^2 + 2*q(5*N+1:6*N).^2); % off diagonal
        q = q./max(1, repmat(norm_q, 6, 1)/alpha0);
       */

    q_x2[pixel_index] = fmaf(sigma, q_x[pixel_index], q_x2[pixel_index]);
    q_y2[pixel_index] = fmaf(sigma, q_y[pixel_index], q_y2[pixel_index]);
    q_xy2[pixel_index] = fmaf(sigma, q_xy[pixel_index], q_xy2[pixel_index]);

    q_z2[pixel_index] = fmaf(sigma, q_z[pixel_index], q_z2[pixel_index]);
    q_xz2[pixel_index] = fmaf(sigma, q_xz[pixel_index], q_xz2[pixel_index]);
    q_yz2[pixel_index] = fmaf(sigma, q_yz[pixel_index], q_yz2[pixel_index]);

    Pixel normalization =
            q_x2[pixel_index] * q_x2[pixel_index] +
            q_y2[pixel_index] * q_y2[pixel_index] +
            2 * q_xy2[pixel_index] * q_xy2[pixel_index] +
            q_z2[pixel_index] * q_z2[pixel_index] +
            2 * q_xz2[pixel_index] * q_xz2[pixel_index] +
            2 * q_yz2[pixel_index] * q_yz2[pixel_index];

    normalization = fmaxf(1, sqrtf(normalization) / alpha0);

    q_x2[pixel_index] /= normalization;
    q_y2[pixel_index] /= normalization;
    q_xy2[pixel_index] /= normalization;

    q_z2[pixel_index] /= normalization;
    q_xz2[pixel_index] /= normalization;
    q_yz2[pixel_index] /= normalization;

    v_previous_z[pixel_index] = v_z[pixel_index];

    v_previous_x[pixel_index] = v_x[pixel_index];
    v_previous_y[pixel_index] = v_y[pixel_index];
}

template<typename Pixel>
void tgv_launch_divergence2_masked(
        Pixel* q_x, Pixel* q_y, Pixel* q_z,
        Pixel* q_xy, Pixel* q_xz, Pixel* q_yz,
        Pixel* q_x2, Pixel* q_y2, Pixel* q_z2,
        Pixel* q_temp,
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

    launch_forward_difference_z_masked(q_xz, q_temp, width_x_height,
        block_dimension, back_grid_dimension, not_back_grid_dimension,
        back_indices, back_indices_count,
        not_back_indices, not_back_indices_count);
    cudaCheckError( cudaDeviceSynchronize() );
    add_masked<<<masked_grid_dimension, block_dimension>>>(
        q_x2, q_temp, masked_indices, masked_indices_count);

    launch_forward_difference_z_masked(q_yz, q_temp, width_x_height,
        block_dimension, back_grid_dimension, not_back_grid_dimension,
        back_indices, back_indices_count,
        not_back_indices, not_back_indices_count);
    cudaCheckError( cudaDeviceSynchronize() );
    add_masked<<<masked_grid_dimension, block_dimension>>>(
        q_y2, q_temp, masked_indices, masked_indices_count);

    launch_forward_difference_z_masked(q_z, q_z2, width_x_height,
        block_dimension, back_grid_dimension, not_back_grid_dimension,
        back_indices, back_indices_count,
        not_back_indices, not_back_indices_count);
    launch_forward_difference_x_masked(q_xz, q_temp,
        block_dimension, right_grid_dimension, not_right_grid_dimension,
        right_indices, right_indices_count,
        not_right_indices, not_right_indices_count);
    cudaCheckError( cudaDeviceSynchronize() );
    add_masked<<<masked_grid_dimension, block_dimension>>>(
        q_z2, q_temp, masked_indices, masked_indices_count);

    launch_forward_difference_y_masked(q_yz, q_temp, width,
        block_dimension, bottom_grid_dimension, not_bottom_grid_dimension,
        bottom_indices, bottom_indices_count,
        not_bottom_indices, not_bottom_indices_count);
    cudaCheckError( cudaDeviceSynchronize() );
    add_masked<<<masked_grid_dimension, block_dimension>>>(
        q_z2, q_temp, masked_indices, masked_indices_count);

    cudaCheckError( cudaDeviceSynchronize() );
}


template<typename Pixel>
__global__ void tgv_kernel_part6_masked(
        Pixel*  v_x, Pixel*  v_y, Pixel* v_z,
        Pixel* q_x2, Pixel* q_y2, Pixel* q_z2,
        Pixel* p_x, Pixel* p_y, Pixel* p_z,
        Pixel* v_previous_x, Pixel* v_previous_y, Pixel* v_previous_z,
        Pixel* v_bar_x, Pixel* v_bar_y, Pixel* v_bar_z,
        const Pixel tau, const Pixel theta,
        Index* indices,  IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    v_x[pixel_index] -= tau * (q_x2[pixel_index] - p_x[pixel_index]);
    v_y[pixel_index] -= tau * (q_y2[pixel_index] - p_y[pixel_index]);

    v_z[pixel_index] -= tau * (q_z2[pixel_index] - p_z[pixel_index]);

    v_bar_z[pixel_index] = v_z[pixel_index] + theta*(v_z[pixel_index] - v_previous_z[pixel_index]);


    v_bar_x[pixel_index] = v_x[pixel_index] + theta*(v_x[pixel_index] - v_previous_x[pixel_index]);
    v_bar_y[pixel_index] = v_y[pixel_index] + theta*(v_y[pixel_index] - v_previous_y[pixel_index]);
}

template<typename Pixel>
__global__ void tgv_kernel_part4_tgv2_l1_masked(
        Pixel* p_x, Pixel* p_y, Pixel* p_z,
        Pixel* u, Pixel* f,
        const Pixel tau_x_lambda, const Pixel tau,
        Pixel* u_previous, const Pixel theta, Pixel* u_bar,
        Index* indices,  IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    u[pixel_index] -= tau * (p_x[pixel_index] + p_y[pixel_index] + p_z[pixel_index]);

    const Pixel u_minus_f = u[pixel_index] - f[pixel_index];
    if(u_minus_f > tau_x_lambda)
        u[pixel_index] -= tau_x_lambda;
    else if(u_minus_f < -tau_x_lambda)
        u[pixel_index] += tau_x_lambda;
    else
        u[pixel_index] = f[pixel_index];

    u_bar[pixel_index] = u[pixel_index] + theta*(u[pixel_index] - u_previous[pixel_index]);
}



#endif // TGV_2_MASKED_COMMON
