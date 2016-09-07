#ifndef TGV_MASKED_COMMON_2D
#define TGV_MASKED_COMMON_2D

#include "tgv_masked_common.cu"

template<typename Pixel>
void tgv_launch_forward_differences_masked_2d(const Pixel* u_bar,
        Pixel* p_x, Pixel* p_y,
        Size width,
        GridDimension block_dimension,
        GridDimension right_grid_dimension,
        GridDimension not_right_grid_dimension,
        GridDimension bottom_grid_dimension,
        GridDimension not_bottom_grid_dimension,
        Index* right_indices, IndexCount right_indices_count,
        Index* not_right_indices, IndexCount not_right_indices_count,
        Index* bottom_indices, IndexCount bottom_indices_count,
        Index* not_bottom_indices, IndexCount not_bottom_indices_count)
{
    launch_forward_difference_x_masked(u_bar, p_x,
        block_dimension, right_grid_dimension, not_right_grid_dimension,
        right_indices, right_indices_count,
        not_right_indices, not_right_indices_count);

    launch_forward_difference_y_masked(u_bar, p_y, width,
        block_dimension, bottom_grid_dimension, not_bottom_grid_dimension,
        bottom_indices, bottom_indices_count,
        not_bottom_indices, not_bottom_indices_count);

    cudaCheckError( cudaDeviceSynchronize() );
}

template<typename Pixel>
void tgv_launch_backward_differences_masked_2d(
        Pixel* p_x, Pixel* p_y,
        const Pixel* p_xx, const Pixel* p_yy,
        Size width,
        GridDimension block_dimension,
        GridDimension left_grid_dimension, GridDimension not_left_grid_dimension,
        GridDimension top_grid_dimension, GridDimension not_top_grid_dimension,
        Index* left_indices, IndexCount left_indices_count,
        Index* not_left_indices, IndexCount not_left_indices_count,
        Index* top_indices, IndexCount top_indices_count,
        Index* not_top_indices, IndexCount not_top_indices_count)
{
    launch_backward_difference_x_masked(p_xx, p_x,
        block_dimension, left_grid_dimension, not_left_grid_dimension,
        left_indices, left_indices_count,
        not_left_indices, not_left_indices_count);

    launch_backward_difference_y_masked(p_yy, p_y, width,
        block_dimension, top_grid_dimension, not_top_grid_dimension,
        top_indices, top_indices_count,
        not_top_indices, not_top_indices_count);

    cudaCheckError( cudaDeviceSynchronize() );
}

template<typename Pixel>
__global__ void tgv_kernel_part2_masked_2d(
        Pixel* p_x, Pixel* p_y,
        Pixel* p_xx, Pixel* p_yy,
        const Pixel sigma, const Pixel alpha1, Pixel* u_previous, Pixel* u,
        Index* indices,  IndexCount indices_count) {
    Index cuda_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cuda_index >= indices_count)
        return;
    Index pixel_index = indices[cuda_index];

    p_xx[pixel_index] = fmaf(sigma, p_x[pixel_index], p_xx[pixel_index]);
    p_yy[pixel_index] = fmaf(sigma, p_y[pixel_index], p_yy[pixel_index]);

    Pixel normalization = sqrtf(p_xx[pixel_index] * p_xx[pixel_index] + p_yy[pixel_index] * p_yy[pixel_index]);

    normalization = fmaxf(1, normalization/alpha1);

    p_xx[pixel_index] /= normalization;
    p_yy[pixel_index] /= normalization;

    u_previous[pixel_index] = u[pixel_index];
}

template<typename Pixel>
void tgv_launch_part1_masked_2d(
          uint width, uint height,
          uint &voxel_count,
          dim3 &block_dimension,

          uint left_indices_count, uint not_left_indices_count,
          uint right_indices_count, uint not_right_indices_count,
          uint top_indices_count, uint not_top_indices_count,
          uint bottom_indices_count, uint not_bottom_indices_count,
          uint masked_indices_count,

          dim3 &left_grid_dimension, dim3 &not_left_grid_dimension,
          dim3 &right_grid_dimension, dim3 &not_right_grid_dimension,
          dim3 &top_grid_dimension, dim3 &not_top_grid_dimension,
          dim3 &bottom_grid_dimension, dim3 &not_bottom_grid_dimension,
          dim3 &masked_grid_dimension, dim3 &all_grid_dimension,
          int cuda_block_dimension = -1)
{
    int cuda_device_count;
    cudaCheckError( cudaGetDeviceCount(&cuda_device_count) );

    cudaDeviceProp device_properties;
    cudaCheckError( cudaGetDeviceProperties(&device_properties, 0) );

 //   printf("found %d cuda devices.\n", cuda_device_count);

    voxel_count = width*height;

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
}

template<typename IndexType>
void freeIndices_2d(IndexType* left_indices,
                 IndexType* not_left_indices,
                 IndexType* right_indices,
                 IndexType* not_right_indices,
                 IndexType* top_indices,
                 IndexType* not_top_indices,
                 IndexType* bottom_indices,
                 IndexType* not_bottom_indices,
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
    cudaFree(masked_indices);
}

#endif //TGV_MASKED_COMMON_2D
