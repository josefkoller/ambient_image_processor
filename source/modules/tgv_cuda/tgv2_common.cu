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

#ifndef TGV_2_COMMON
#define TGV_2_COMMON

#include "cuda_helper.cuh"

#include "tgv_common.cu"


template<typename Pixel>
__global__ void addAndHalf(
        Pixel* v_xy, Pixel* v_yx, Pixel* q_xy,
        uint width, uint height, uint depth) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width*height*depth)
        return;

    q_xy[index] = (v_xy[index] + v_yx[index]) * 0.5;
}


template<typename Pixel>
__global__ void add(
        Pixel* q_xx, Pixel* q_yx, Pixel* q_x2,
        uint width, uint height, uint depth) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width*height*depth)
        return;

    q_x2[index] = q_xx[index] + q_yx[index];
}

template<typename Pixel>
void tgv_launch_part22(
        uint voxel_count,
        Pixel** v_bar_x, Pixel** v_bar_y, Pixel** v_bar_z,
        Pixel** v_previous_x, Pixel** v_previous_y, Pixel** v_previous_z,
        Pixel** v_x, Pixel** v_y, Pixel** v_z,
        Pixel** q_x, Pixel** q_y, Pixel** q_z,
        Pixel** q_xy, Pixel** q_xz, Pixel** q_yz,
        Pixel** q_x2, Pixel** q_y2, Pixel** q_z2,
        Pixel** q_xy2, Pixel** q_xz2, Pixel** q_yz2, Pixel** q_temp)
{
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMalloc(v_x, size) )
    cudaCheckError( cudaMalloc(v_y, size) )
    cudaCheckError( cudaMalloc(v_bar_x, size) )
    cudaCheckError( cudaMalloc(v_bar_y, size) )
    cudaCheckError( cudaMalloc(v_previous_x, size) )
    cudaCheckError( cudaMalloc(v_previous_y, size) )

    cudaCheckError( cudaMalloc(q_x, size) )
    cudaCheckError( cudaMalloc(q_y, size) )
    cudaCheckError( cudaMalloc(q_xy, size) )
    cudaCheckError( cudaMalloc(q_x2, size) )
    cudaCheckError( cudaMalloc(q_y2, size) )
    cudaCheckError( cudaMalloc(q_xy2, size) )
    cudaCheckError( cudaMalloc(q_temp, size) )

    cudaCheckError( cudaMalloc(v_z, size) )
    cudaCheckError( cudaMalloc(v_bar_z, size) )
    cudaCheckError( cudaMalloc(v_previous_z, size) )

    cudaCheckError( cudaMalloc(q_z, size) )
    cudaCheckError( cudaMalloc(q_xz, size) )
    cudaCheckError( cudaMalloc(q_yz, size) )

    cudaCheckError( cudaMalloc(q_z2, size) )
    cudaCheckError( cudaMalloc(q_xz2, size) )
    cudaCheckError( cudaMalloc(q_yz2, size) )
}

template<typename Pixel>
__global__ void tgv_kernel_part22(
        Pixel* v_bar_x, Pixel* v_bar_y, Pixel* v_bar_z,
        Pixel* p_x, Pixel* p_y, Pixel* p_z,
        Pixel* p_xx, Pixel* p_yy, Pixel* p_zz,
        const Pixel sigma, const Pixel alpha1, Pixel* u_previous, Pixel* u,
        const uint width, const uint height, const uint depth) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
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
    p_zz[index] = fmaf(sigma, p_z[index] - v_bar_z[index], p_zz[index]);

    Pixel normalization = norm3df(p_xx[index], p_yy[index], p_zz[index]);

    normalization = fmaxf(1, normalization/alpha1);

    p_xx[index] /= normalization;
    p_yy[index] /= normalization;
    p_zz[index] /= normalization;

    u_previous[index] = u[index];
}

template<typename Pixel>
void tgv_launch_part32(uint depth,
                       Pixel* v_bar_x, Pixel* v_bar_y, Pixel* v_bar_z,
                       Pixel* v_previous_x, Pixel* v_previous_y, Pixel* v_previous_z,
                       Pixel* v_x, Pixel* v_y, Pixel* v_z,
                       Pixel* q_x, Pixel* q_y, Pixel* q_z,
                       Pixel* q_xy, Pixel* q_xz, Pixel* q_yz,
                       Pixel* q_x2, Pixel* q_y2, Pixel* q_z2,
                       Pixel* q_xy2, Pixel* q_xz2, Pixel* q_yz2, Pixel* q_temp)
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

    cudaFree(v_z);
    cudaFree(v_bar_z);
    cudaFree(v_previous_z);

    cudaFree(q_z);
    cudaFree(q_xz);
    cudaFree(q_yz);

    cudaFree(q_z2);
    cudaFree(q_xz2);
    cudaFree(q_yz2);
}



template<typename Pixel>
void tgv_launch_gradient2(
        Pixel* v_bar_x, Pixel* v_bar_y, Pixel* v_bar_z,
        Pixel* q_x, Pixel* q_y, Pixel* q_z,
        Pixel* q_xy, Pixel* q_xz, Pixel* q_yz,  Pixel* q_temp,
        uint width, uint height, uint depth,
        dim3 block_dimension,
        dim3 grid_dimension,
        dim3 grid_dimension_x,
        dim3 grid_dimension_y,
        dim3 grid_dimension_z)
{
    backward_difference_x<<<grid_dimension_x, block_dimension>>>(
          v_bar_x, q_x, width, height, depth);

    backward_difference_y<<<grid_dimension_y, block_dimension>>>(
          v_bar_y, q_y, width, height, depth);

    backward_difference_x<<<grid_dimension_x, block_dimension>>>(
          v_bar_y, q_xy, width, height, depth);
    backward_difference_y<<<grid_dimension_y, block_dimension>>>(
          v_bar_x, q_temp, width, height, depth);
    addAndHalf<<<grid_dimension, block_dimension>>>(
            q_xy, q_temp, q_xy,
            width, height, depth);

    backward_difference_z<<<grid_dimension_z, block_dimension>>>(
          v_bar_z, q_z, width, height, depth);

    backward_difference_x<<<grid_dimension_x, block_dimension>>>(
          v_bar_z, q_xz, width, height, depth);
    backward_difference_z<<<grid_dimension_z, block_dimension>>>(
          v_bar_x, q_temp, width, height, depth);
    addAndHalf<<<grid_dimension, block_dimension>>>(
            q_xz, q_temp, q_xz,
            width, height, depth);

    backward_difference_y<<<grid_dimension_y, block_dimension>>>(
          v_bar_z, q_yz, width, height, depth);
    backward_difference_z<<<grid_dimension_z, block_dimension>>>(
          v_bar_y, q_temp, width, height, depth);
    addAndHalf<<<grid_dimension, block_dimension>>>(
            q_yz, q_temp, q_yz,
            width, height, depth);

    cudaCheckError( cudaDeviceSynchronize() );
}

template<typename Pixel>
__global__ void tgv_kernel_part5(
        Pixel* v_x,Pixel* v_y,Pixel* v_z,
        Pixel* v_previous_x, Pixel* v_previous_y, Pixel* v_previous_z,
        Pixel* q_x,Pixel* q_y,Pixel* q_z,
        Pixel* q_xy,Pixel* q_xz,Pixel* q_yz,
        Pixel* q_x2, Pixel* q_y2, Pixel* q_z2,
        Pixel* q_xy2, Pixel* q_xz2, Pixel* q_yz2,
        const Pixel sigma, const Pixel alpha0,
        const uint width, const uint height, const uint depth) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
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

    q_z2[index] = fmaf(sigma, q_z[index], q_z2[index]);
    q_xz2[index] = fmaf(sigma, q_xz[index], q_xz2[index]);
    q_yz2[index] = fmaf(sigma, q_yz[index], q_yz2[index]);

    Pixel normalization =
            q_x2[index] * q_x2[index] +
            q_y2[index] * q_y2[index] +
            2 * q_xy2[index] * q_xy2[index] +
            q_z2[index] * q_z2[index] +
            2 * q_xz2[index] * q_xz2[index] +
            2 * q_yz2[index] * q_yz2[index];

    normalization = fmaxf(1, sqrtf(normalization) / alpha0);

    q_x2[index] /= normalization;
    q_y2[index] /= normalization;
    q_xy2[index] /= normalization;

    q_z2[index] /= normalization;
    q_xz2[index] /= normalization;
    q_yz2[index] /= normalization;

    v_previous_z[index] = v_z[index];

    v_previous_x[index] = v_x[index];
    v_previous_y[index] = v_y[index];
}

template<typename Pixel>
void tgv_launch_divergence2(
        Pixel* q_x, Pixel* q_y, Pixel* q_z,
        Pixel* q_xy, Pixel* q_xz, Pixel* q_yz,
        Pixel* q_x2, Pixel* q_y2, Pixel* q_z2,
        Pixel* q_temp,
        uint width, uint height, uint depth,
        dim3 block_dimension,
        dim3 grid_dimension,
        dim3 grid_dimension_x,
        dim3 grid_dimension_y,
        dim3 grid_dimension_z)
{
    forward_difference_x<<<grid_dimension_x, block_dimension>>>(
      q_x, q_x2, width, height, depth);
    forward_difference_y<<<grid_dimension_y, block_dimension>>>(
      q_xy, q_temp, width, height, depth);
    add<<<grid_dimension, block_dimension>>>(q_x2, q_temp, q_x2, width, height, depth);

    forward_difference_y<<<grid_dimension_y, block_dimension>>>(
      q_y, q_y2, width, height, depth);
    forward_difference_x<<<grid_dimension_x, block_dimension>>>(
      q_xy, q_temp, width, height, depth);
    add<<<grid_dimension, block_dimension>>>(q_y2, q_temp, q_y2, width, height, depth);

    forward_difference_z<<<grid_dimension_z, block_dimension>>>(
      q_xz, q_temp, width, height, depth);
    add<<<grid_dimension, block_dimension>>>(q_x2, q_temp, q_x2, width, height, depth);

    forward_difference_z<<<grid_dimension_z, block_dimension>>>(
      q_yz, q_temp, width, height, depth);
    add<<<grid_dimension, block_dimension>>>(q_y2, q_temp, q_y2, width, height, depth);

    forward_difference_z<<<grid_dimension_z, block_dimension>>>(
      q_z, q_z2, width, height, depth);
    forward_difference_x<<<grid_dimension_x, block_dimension>>>(
      q_xz, q_temp, width, height, depth);
    add<<<grid_dimension, block_dimension>>>(q_z2, q_temp, q_z2, width, height, depth);
    forward_difference_y<<<grid_dimension_y, block_dimension>>>(
      q_yz, q_temp, width, height, depth);
    add<<<grid_dimension, block_dimension>>>(q_z2, q_temp, q_z2, width, height, depth);
    cudaCheckError( cudaDeviceSynchronize() );
}


template<typename Pixel>
__global__ void tgv_kernel_part6(
        Pixel*  v_x, Pixel*  v_y, Pixel* v_z,
        Pixel* q_x2, Pixel* q_y2, Pixel* q_z2,
        Pixel* p_x, Pixel* p_y, Pixel* p_z,
        Pixel* v_previous_x, Pixel* v_previous_y, Pixel* v_previous_z,
        Pixel* v_bar_x, Pixel* v_bar_y, Pixel* v_bar_z,
        const Pixel tau, const Pixel theta,
        const uint width, const uint height, const uint depth)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    v_x[index] -= tau * (q_x2[index] - p_x[index]);
    v_y[index] -= tau * (q_y2[index] - p_y[index]);

    v_z[index] -= tau * (q_z2[index] - p_z[index]);

    v_bar_z[index] = v_z[index] + theta*(v_z[index] - v_previous_z[index]);


    v_bar_x[index] = v_x[index] + theta*(v_x[index] - v_previous_x[index]);
    v_bar_y[index] = v_y[index] + theta*(v_y[index] - v_previous_y[index]);

    /*
     *  Matlab Code:
            v = v - tau * (nabla_second_t * q - p);
            v_bar = v + theta*(v - v_old);
    */
}

template<typename Pixel>
__global__ void zeroInit2(
        Pixel* p_x, Pixel* p_y, Pixel* p_z,
        Pixel* p_xx, Pixel* q_xz, Pixel* p_zz,
        uint voxel_count) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= voxel_count)
        return;

    p_x[index] = p_y[index] =
    p_xx[index] = 0;

    p_z[index] = p_zz[index] = q_xz[index] = 0;
}

template<typename Pixel>
bool tgv2_deshade_iteration_callback(
        uint iteration_index, const uint iteration_count, const int paint_iteration_interval,
        Pixel* u, Pixel* v_x, Pixel* v_y, Pixel* v_z,
        std::function<bool(uint iteration_index, uint iteration_count,
                           Pixel* u, Pixel* v_x, Pixel* v_y, Pixel* v_z)> iteration_callback,
        const uint voxel_count)
{

    if(paint_iteration_interval > 0 && iteration_index > 0 &&
            iteration_index % paint_iteration_interval == 0 &&
            iteration_callback != nullptr) {
        printf("iteration %d / %d \n", iteration_index, iteration_count);

        Pixel* u_host = new Pixel[voxel_count];
        Pixel* v_x_host = new Pixel[voxel_count];
        Pixel* v_y_host = new Pixel[voxel_count];
        Pixel* v_z_host = new Pixel[voxel_count];
        auto size = sizeof(Pixel) * voxel_count;
        cudaCheckError( cudaMemcpy(u_host, u, size, cudaMemcpyDeviceToHost) );
        cudaCheckError( cudaMemcpy(v_x_host, v_x, size, cudaMemcpyDeviceToHost) );
        cudaCheckError( cudaMemcpy(v_y_host, v_y, size, cudaMemcpyDeviceToHost) );
        cudaCheckError( cudaMemcpy(v_z_host, v_z, size, cudaMemcpyDeviceToHost) );
        bool stop = iteration_callback(iteration_index, iteration_count, u_host,
                                       v_x_host, v_y_host, v_z_host);
        delete[] u_host;
        delete[] v_x_host;
        delete[] v_y_host;
        delete[] v_z_host;
        return stop;
    }
    else
    {
        return false;
    }
}

#endif // TGV_2_COMMON
