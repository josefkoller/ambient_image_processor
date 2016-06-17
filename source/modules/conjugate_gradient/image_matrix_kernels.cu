
#include "cuda_helper.cuh"

typedef const unsigned int Dimension;

template<typename Pixel>
__global__  void set_zeros_kernel(Pixel* elements, Dimension element_count)
{
    Dimension index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= element_count)
        return;

    elements[index] = 0;
}

template<typename Pixel>
void set_zeros_kernel_launch(Pixel* elements, Dimension element_count,
                             dim3 block_dimension, dim3 grid_dimension)
{
    set_zeros_kernel<<<grid_dimension, block_dimension>>>(elements, element_count);
    cudaCheckError( cudaDeviceSynchronize() );
}


template<typename Pixel>
__global__  void transposed_kernel(Pixel* elements, Dimension element_count, Dimension voxel_count,
                                   Pixel* transposed_elements)
{
    Dimension index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= element_count)
        return;

    Dimension y = index / voxel_count;
    Dimension x = index - y * voxel_count;
    Dimension target_index = x * voxel_count + y;

    transposed_elements[target_index] = elements[index];
}

template<typename Pixel>
void transposed_kernel_launch(Pixel* elements, Dimension element_count, Dimension voxel_count,
                         Pixel* transposed_elements,
                         dim3 block_dimension, dim3 grid_dimension)
{
    transposed_kernel<<<grid_dimension, block_dimension>>>(elements, element_count, voxel_count, transposed_elements);
    cudaCheckError( cudaDeviceSynchronize() );
}

template<typename Pixel>
__global__  void add_kernel(Pixel* elements, Dimension element_count, Pixel* elements2, Pixel* result)
{
    Dimension index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= element_count)
        return;

    result[index] = elements[index] + elements2[index];
}

template<typename Pixel>
void add_kernel_launch(Pixel* elements, Dimension element_count,
                       Pixel* elements2,
                       Pixel* result,
                       dim3 block_dimension, dim3 grid_dimension)
{
    add_kernel<<<grid_dimension, block_dimension>>>(elements, element_count, elements2, result);
    cudaCheckError( cudaDeviceSynchronize() );
}

template<typename Pixel>
__global__  void multiply_kernel(Pixel* elements, Dimension element_count,
                                 Dimension voxel_count, Pixel* elements2, Pixel* result)
{
    Dimension index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= element_count)
        return;

    Dimension y = index / voxel_count;
    Dimension y_offset = y * voxel_count;
    Dimension x = index - y_offset;

    result[index] = 0;
    for(int i = 0; i < voxel_count; i++)
    {
        result[index] += elements[y_offset + i] * elements[i * voxel_count + x];
    }
}

template<typename Pixel>
void multiply_kernel_launch(Pixel* elements, Dimension element_count, Dimension voxel_count,
                       Pixel* elements2,
                       Pixel* result,
                       dim3 block_dimension, dim3 grid_dimension)
{
    multiply_kernel<<<grid_dimension, block_dimension>>>(elements, element_count, voxel_count, elements2, result);
    cudaCheckError( cudaDeviceSynchronize() );
}


template void multiply_kernel_launch(float* elements, Dimension element_count, Dimension voxel_count,
                       float* elements2, float* result, dim3 block_dimension, dim3 grid_dimension);
template void add_kernel_launch(float* elements, Dimension element_count,
                       float* elements2, float* result, dim3 block_dimension, dim3 grid_dimension);
template void transposed_kernel_launch(float* elements, Dimension element_count, Dimension voxel_count,
                         float* transposed_elements,
                         dim3 block_dimension, dim3 grid_dimension);
template void set_zeros_kernel_launch(float* elements, Dimension element_count,
                             dim3 block_dimension, dim3 grid_dimension);

template void multiply_kernel_launch(double* elements, Dimension element_count, Dimension voxel_count,
                       double* elements2, double* result, dim3 block_dimension, dim3 grid_dimension);
template void add_kernel_launch(double* elements, Dimension element_count,
                       double* elements2, double* result, dim3 block_dimension, dim3 grid_dimension);
template void transposed_kernel_launch(double* elements, Dimension element_count, Dimension voxel_count,
                         double* transposed_elements,
                         dim3 block_dimension, dim3 grid_dimension);
template void set_zeros_kernel_launch(double* elements, Dimension element_count,
                             dim3 block_dimension, dim3 grid_dimension);

