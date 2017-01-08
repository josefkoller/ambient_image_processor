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

#ifndef TGV_MASKED_COMMON
#define TGV_MASKED_COMMON

#include "cuda.h"
#include <functional>
#include <vector>

#include "tgv_common.cu"
#include "cuda_helper.cuh"

typedef uint NotConstIndex;
typedef const NotConstIndex Index;

typedef uint NotConstIndexCount;
typedef const NotConstIndexCount IndexCount;

typedef const uint Size; // width, height, depth
typedef const dim3 GridDimension;

typedef const std::vector<uint>& IndexVector;

template<typename IndexType>
void copyIndicesToDevice(const std::vector<IndexType>& host_indices,
                         IndexType** indices, IndexType& indices_count)
{
    indices_count = host_indices.size();

    if(indices_count == 0)
    {
        *indices = nullptr;
        return;
    }

    size_t size = sizeof(IndexType) * indices_count;
    cudaCheckError( cudaMalloc(indices, size) );

    auto host_indices_array = new IndexType[indices_count];
    for(uint i = 0; i < indices_count; i++)
    {
        host_indices_array[i] = host_indices[i];
    }
    cudaCheckError( cudaMemcpy(*indices, host_indices_array, size, cudaMemcpyHostToDevice) );
}

template<typename IndexType>
void freeIndices(IndexType* left_indices,
                 IndexType* not_left_indices,
                 IndexType* right_indices,
                 IndexType* not_right_indices,
                 IndexType* top_indices,
                 IndexType* not_top_indices,
                 IndexType* bottom_indices,
                 IndexType* not_bottom_indices,
                 IndexType* front_indices,
                 IndexType* not_front_indices,
                 IndexType* back_indices,
                 IndexType* not_back_indices,
                 IndexType* masked_indices
                 )
{
    cudaFree(left_indices);
    cudaFree(not_left_indices);
    cudaFree(right_indices);
    cudaFree(not_right_indices);
    cudaFree(top_indices);
    cudaFree(not_top_indices);
    cudaFree(bottom_indices);
    cudaFree(not_bottom_indices);
    cudaFree(front_indices);
    cudaFree(not_front_indices);
    cudaFree(back_indices);
    cudaFree(not_back_indices);
    cudaFree(masked_indices);
}

template<typename Pixel>
using DeshadeIterationCallback = std::function<bool(uint iteration_index, uint iteration_count,
    Pixel* u, Pixel* v_x, Pixel* v_y, Pixel* v_z)>;

template<typename Pixel>
__global__ void set_zero_masked(Pixel* p_x, Index* indices, IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    p_x[pixel_index] = 0;
}

template<typename Pixel>
__global__ void forward_difference_x_masked(
        const Pixel* u_bar, Pixel* p,
        Index* indices,  IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    p[pixel_index] = u_bar[pixel_index + 1] - u_bar[pixel_index];
}

template<typename Pixel>
__global__ void forward_difference_y_masked(
        const Pixel* u_bar, Pixel* p, Size width,
        Index* indices,  IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    p[pixel_index] = u_bar[pixel_index + width] - u_bar[pixel_index];
}

template<typename Pixel>
__global__ void forward_difference_z_masked(
        const Pixel* u_bar, Pixel* p, Size width_x_height,
        Index* indices,  IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    p[pixel_index] = u_bar[pixel_index + width_x_height] - u_bar[pixel_index];
}

template<typename Pixel>
__global__ void swap_sign_masked(const Pixel* u, Pixel* p, Index* indices, IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    p[pixel_index] = - u[pixel_index];
}

template<typename Pixel>
__global__ void backward_difference_x_masked(
        const Pixel* u_bar, Pixel* p,
        Index* indices,  IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    p[pixel_index] =  - u_bar[pixel_index] + u_bar[pixel_index - 1];
}

template<typename Pixel>
__global__ void backward_difference_y_masked(
        const Pixel* u_bar, Pixel* p, Size width,
        Index* indices,  IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    p[pixel_index] =  - u_bar[pixel_index] + u_bar[pixel_index - width];
}

template<typename Pixel>
__global__ void backward_difference_z_masked(
        const Pixel* u_bar, Pixel* p, Size width_x_height,
        Index* indices,  IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    p[pixel_index] =  - u_bar[pixel_index] + u_bar[pixel_index - width_x_height];
}

template<typename Pixel>
void launch_forward_difference_x_masked(const Pixel* u_bar, Pixel* p_x,
        GridDimension block_dimension,
        GridDimension right_grid_dimension,
        GridDimension not_right_grid_dimension,
        Index* right_indices, IndexCount right_indices_count,
        Index* not_right_indices, IndexCount not_right_indices_count)
{
    if(not_right_indices_count > 0)
        forward_difference_x_masked<<<not_right_grid_dimension, block_dimension>>>(
              u_bar, p_x, not_right_indices, not_right_indices_count);

    if(right_indices_count > 0)
        set_zero_masked<<<right_grid_dimension, block_dimension>>>(
              p_x, right_indices, right_indices_count);
}

template<typename Pixel>
void launch_forward_difference_y_masked(const Pixel* u_bar, Pixel* p_y,
        Size width,
        GridDimension block_dimension,
        GridDimension bottom_grid_dimension,
        GridDimension not_bottom_grid_dimension,
        Index* bottom_indices, IndexCount bottom_indices_count,
        Index* not_bottom_indices, IndexCount not_bottom_indices_count)
{
    if(not_bottom_indices_count > 0)
        forward_difference_y_masked<<<not_bottom_grid_dimension, block_dimension>>>(
              u_bar, p_y, width, not_bottom_indices, not_bottom_indices_count);

    if(bottom_indices_count > 0)
        set_zero_masked<<<bottom_grid_dimension, block_dimension>>>(
              p_y, bottom_indices, bottom_indices_count);
}

template<typename Pixel>
void launch_forward_difference_z_masked(const Pixel* u_bar, Pixel* p_z,
        Size width_x_height,
        GridDimension block_dimension,
        GridDimension back_grid_dimension,
        GridDimension not_back_grid_dimension,
        Index* back_indices, IndexCount back_indices_count,
        Index* not_back_indices, IndexCount not_back_indices_count)
{
    if(not_back_indices_count > 0)
        forward_difference_z_masked<<<not_back_grid_dimension, block_dimension>>>(
              u_bar, p_z, width_x_height, not_back_indices, not_back_indices_count);

    if(back_indices_count > 0)
        set_zero_masked<<<back_grid_dimension, block_dimension>>>(
              p_z, back_indices, back_indices_count);
}

template<typename Pixel>
void tgv_launch_forward_differences_masked(const Pixel* u_bar,
        Pixel* p_x, Pixel* p_y, Pixel* p_z,
        Size width, Size width_x_height,
        GridDimension block_dimension,
        GridDimension right_grid_dimension,
        GridDimension not_right_grid_dimension,
        GridDimension bottom_grid_dimension,
        GridDimension not_bottom_grid_dimension,
        GridDimension back_grid_dimension,
        GridDimension not_back_grid_dimension,
        Index* right_indices, IndexCount right_indices_count,
        Index* not_right_indices, IndexCount not_right_indices_count,
        Index* bottom_indices, IndexCount bottom_indices_count,
        Index* not_bottom_indices, IndexCount not_bottom_indices_count,
        Index* back_indices, IndexCount back_indices_count,
        Index* not_back_indices, IndexCount not_back_indices_count)
{
    launch_forward_difference_x_masked(u_bar, p_x,
        block_dimension, right_grid_dimension, not_right_grid_dimension,
        right_indices, right_indices_count,
        not_right_indices, not_right_indices_count);

    launch_forward_difference_y_masked(u_bar, p_y, width,
        block_dimension, bottom_grid_dimension, not_bottom_grid_dimension,
        bottom_indices, bottom_indices_count,
        not_bottom_indices, not_bottom_indices_count);

    launch_forward_difference_z_masked(u_bar, p_z, width_x_height,
        block_dimension, back_grid_dimension, not_back_grid_dimension,
        back_indices, back_indices_count,
        not_back_indices, not_back_indices_count);

    cudaCheckError( cudaDeviceSynchronize() );
}

template<typename Pixel>
void launch_backward_difference_x_masked(const Pixel* p_xx, Pixel* p_x,
        GridDimension block_dimension,
        GridDimension left_grid_dimension,
        GridDimension not_left_grid_dimension,
        Index* left_indices, IndexCount left_indices_count,
        Index* not_left_indices, IndexCount not_left_indices_count)
{
    if(not_left_indices_count > 0)
        backward_difference_x_masked<<<not_left_grid_dimension, block_dimension>>>(
          p_xx, p_x, not_left_indices, not_left_indices_count);

    if(left_indices_count > 0)
        swap_sign_masked<<<left_grid_dimension, block_dimension>>>(
          p_xx, p_x, left_indices, left_indices_count);
}

template<typename Pixel>
void launch_backward_difference_y_masked(const Pixel* p_yy, Pixel* p_y, Size width,
        GridDimension block_dimension,
        GridDimension top_grid_dimension,
        GridDimension not_top_grid_dimension,
        Index* top_indices, IndexCount top_indices_count,
        Index* not_top_indices, IndexCount not_top_indices_count)
{
    if(not_top_indices_count > 0)
        backward_difference_y_masked<<<not_top_grid_dimension, block_dimension>>>(
          p_yy, p_y, width, not_top_indices, not_top_indices_count);

    if(top_indices_count > 0)
        swap_sign_masked<<<top_grid_dimension, block_dimension>>>(
          p_yy, p_y, top_indices, top_indices_count);
}

template<typename Pixel>
void launch_backward_difference_z_masked(const Pixel* p_zz, Pixel* p_z, Size width_x_height,
        GridDimension block_dimension,
        GridDimension front_grid_dimension,
        GridDimension not_front_grid_dimension,
        Index* front_indices, IndexCount front_indices_count,
        Index* not_front_indices, IndexCount not_front_indices_count)
{
    if(not_front_indices_count > 0)
        backward_difference_z_masked<<<not_front_grid_dimension, block_dimension>>>(
          p_zz, p_z, width_x_height, not_front_indices, not_front_indices_count);

    if(front_indices_count > 0)
        swap_sign_masked<<<front_grid_dimension, block_dimension>>>(
          p_zz, p_z, front_indices, front_indices_count);
}

template<typename Pixel>
void tgv_launch_backward_differences_masked(
        Pixel* p_x, Pixel* p_y, Pixel* p_z,
        const Pixel* p_xx, const Pixel* p_yy, const Pixel* p_zz,
        Size width, Size width_x_height,
        GridDimension block_dimension,
        GridDimension left_grid_dimension, GridDimension not_left_grid_dimension,
        GridDimension top_grid_dimension, GridDimension not_top_grid_dimension,
        GridDimension front_grid_dimension, GridDimension not_front_grid_dimension,
        Index* left_indices, IndexCount left_indices_count,
        Index* not_left_indices, IndexCount not_left_indices_count,
        Index* top_indices, IndexCount top_indices_count,
        Index* not_top_indices, IndexCount not_top_indices_count,
        Index* front_indices, IndexCount front_indices_count,
        Index* not_front_indices, IndexCount not_front_indices_count)
{
    launch_backward_difference_x_masked(p_xx, p_x,
        block_dimension, left_grid_dimension, not_left_grid_dimension,
        left_indices, left_indices_count,
        not_left_indices, not_left_indices_count);

    launch_backward_difference_y_masked(p_yy, p_y, width,
        block_dimension, top_grid_dimension, not_top_grid_dimension,
        top_indices, top_indices_count,
        not_top_indices, not_top_indices_count);

    launch_backward_difference_z_masked(p_zz, p_z, width_x_height,
        block_dimension, front_grid_dimension, not_front_grid_dimension,
        front_indices, front_indices_count,
        not_front_indices, not_front_indices_count);

    cudaCheckError( cudaDeviceSynchronize() );
}

template<typename Pixel>
__global__ void tgv_kernel_part2_masked(
        Pixel* p_x, Pixel* p_y, Pixel* p_z,
        Pixel* p_xx, Pixel* p_yy, Pixel* p_zz,
        const Pixel sigma, const Pixel alpha1, Pixel* u_previous, Pixel* u,
        Index* indices,  IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    /*
    * Matlab Code:
    p = p + sigma*nabla*u_bar;
    norm_p = sqrt(p(1:N).^2 + p(N+1:2*N).^2);
    p = p./max(1,[norm_p; norm_p]);

    u_old = u;
    */

    p_xx[pixel_index] = fmaf(sigma, p_x[pixel_index], p_xx[pixel_index]);
    p_yy[pixel_index] = fmaf(sigma, p_y[pixel_index], p_yy[pixel_index]);
    p_zz[pixel_index] = fmaf(sigma, p_z[pixel_index], p_zz[pixel_index]);

    Pixel normalization = norm3df(p_xx[pixel_index], p_yy[pixel_index], p_zz[pixel_index]);

    normalization = fmaxf(1, normalization/alpha1);

    p_xx[pixel_index] /= normalization;
    p_yy[pixel_index] /= normalization;
    p_zz[pixel_index] /= normalization;

    u_previous[pixel_index] = u[pixel_index];
}

template<typename Pixel>
void tgv_launch_part1_masked(
          uint width, uint height, uint depth,
          uint &voxel_count,
          dim3 &block_dimension,

          uint left_indices_count, uint not_left_indices_count,
          uint right_indices_count, uint not_right_indices_count,
          uint top_indices_count, uint not_top_indices_count,
          uint bottom_indices_count, uint not_bottom_indices_count,
          uint front_indices_count, uint not_front_indices_count,
          uint back_indices_count, uint not_back_indices_count,
          uint masked_indices_count,

          dim3 &left_grid_dimension, dim3 &not_left_grid_dimension,
          dim3 &right_grid_dimension, dim3 &not_right_grid_dimension,
          dim3 &top_grid_dimension, dim3 &not_top_grid_dimension,
          dim3 &bottom_grid_dimension, dim3 &not_bottom_grid_dimension,
          dim3 &front_grid_dimension, dim3 &not_front_grid_dimension,
          dim3 &back_grid_dimension, dim3 &not_back_grid_dimension,
          dim3 &masked_grid_dimension, dim3 &all_grid_dimension,
          int cuda_block_dimension = -1)
{
    int cuda_device_count;
    cudaCheckError( cudaGetDeviceCount(&cuda_device_count) );

    cudaDeviceProp device_properties;
    cudaCheckError( cudaGetDeviceProperties(&device_properties, 0) );

 //   printf("found %d cuda devices.\n", cuda_device_count);

    voxel_count = width*height*depth;

    if(cuda_block_dimension < 0)
        cuda_block_dimension = CUDA_BLOCK_DIMENSON;

    if(cuda_block_dimension > device_properties.maxThreadsPerBlock) {
        cuda_block_dimension = device_properties.maxThreadsPerBlock;
        printf("setting the maximum block dimension: %d \n", cuda_block_dimension);
    }
    //printf("block dimension3: %d \n", cuda_block_dimension);
    block_dimension = dim3(cuda_block_dimension);

    auto grid_dimension = [=](uint count) {
      return dim3((count + cuda_block_dimension - 1) / cuda_block_dimension);
    };

    masked_grid_dimension = grid_dimension(masked_indices_count);
    all_grid_dimension = grid_dimension(voxel_count);

    left_grid_dimension = grid_dimension(left_indices_count);
    not_left_grid_dimension = grid_dimension(not_left_indices_count);
    right_grid_dimension = grid_dimension(right_indices_count);
    not_right_grid_dimension = grid_dimension(not_right_indices_count);
    top_grid_dimension = grid_dimension(top_indices_count);
    not_top_grid_dimension = grid_dimension(not_top_indices_count);
    bottom_grid_dimension = grid_dimension(bottom_indices_count);
    not_bottom_grid_dimension = grid_dimension(not_bottom_indices_count);
    front_grid_dimension = grid_dimension(front_indices_count);
    not_front_grid_dimension = grid_dimension(not_front_indices_count);
    back_grid_dimension = grid_dimension(back_indices_count);
    not_back_grid_dimension = grid_dimension(not_back_indices_count);
}

#endif //TGV_MASKED_COMMON
