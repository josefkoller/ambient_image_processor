
#include "cuda_helper.cuh"

typedef const unsigned int Dimension;

template<typename Pixel>
__global__ void multiply_kernel(
        Pixel* image1, Pixel* image2, Pixel* temp,
        Dimension voxel_count)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= voxel_count)
        return;

    temp[index] = image1[index] * image2[index];
}

template<typename Pixel>
void image_multiply_kernel_launch(Pixel* image1, Pixel* image2, Pixel* temp, Dimension voxel_count)
{
    dim3 block_dimension = dim3(CUDA_BLOCK_DIMENSON);
    dim3 grid_dimension = dim3((voxel_count + block_dimension.x - 1) / block_dimension.x);

    multiply_kernel<<<grid_dimension, block_dimension>>>(
         image1, image2, temp, voxel_count);
    cudaCheckError( cudaDeviceSynchronize() );
}


template<typename Pixel>
__global__ void image_multiply_matrix_kernel(
        Pixel* matrix_elements, Pixel* vector, Pixel* result,
        Dimension voxel_count)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= voxel_count)
        return;

    result[index] = 0;
    for(int i = 0; i < voxel_count; i++)
    {
        Dimension matrix_index = index * voxel_count + i;
        result[index] += matrix_elements[matrix_index] * vector[index];
    }
}

template<typename Pixel>
void image_multiply_matrix_kernel_launch(Pixel* matrix_elements, Pixel* vector, Pixel* result,
                                         Dimension voxel_count)
{
    dim3 block_dimension = dim3(CUDA_BLOCK_DIMENSON);
    dim3 grid_dimension = dim3((voxel_count + block_dimension.x - 1) / block_dimension.x);

    image_multiply_matrix_kernel<<<grid_dimension, block_dimension>>>(
         matrix_elements, vector, result, voxel_count);
    cudaCheckError( cudaDeviceSynchronize() );
}

template<typename Pixel>
__global__ void image_add_kernel(
        Pixel* image1, Pixel* image2, Pixel* result,
        Dimension voxel_count)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= voxel_count)
        return;

    result[index] = image1[index] + image2[index];
}
template<typename Pixel>
void image_add_kernel_launch(Pixel* image1, Pixel* image2, Pixel* result,
                             Dimension voxel_count)
{
    dim3 block_dimension = dim3(CUDA_BLOCK_DIMENSON);
    dim3 grid_dimension = dim3((voxel_count + block_dimension.x - 1) / block_dimension.x);

    image_add_kernel<<<grid_dimension, block_dimension>>>(
         image1, image2, result, voxel_count);
    cudaCheckError( cudaDeviceSynchronize() );
}


template<typename Pixel>
__global__ void image_subtract_kernel(
        Pixel* image1, Pixel* image2, Pixel* result,
        Dimension voxel_count)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= voxel_count)
        return;

    result[index] = image1[index] - image2[index];
}
template<typename Pixel>
void image_subtract_kernel_launch(Pixel* image1, Pixel* image2, Pixel* result,
                             Dimension voxel_count)
{
    dim3 block_dimension = dim3(CUDA_BLOCK_DIMENSON);
    dim3 grid_dimension = dim3((voxel_count + block_dimension.x - 1) / block_dimension.x);

    image_subtract_kernel<<<grid_dimension, block_dimension>>>(
         image1, image2, result, voxel_count);
    cudaCheckError( cudaDeviceSynchronize() );
}


template<typename Pixel>
__global__ void image_scale_kernel(
        Pixel* image1, Pixel factor, Pixel* result,
        Dimension voxel_count)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= voxel_count)
        return;

    result[index] = image1[index] * factor;
}
template<typename Pixel>
void image_scale_kernel_launch(Pixel* image1, Pixel factor, Pixel* result,
                             Dimension voxel_count)
{
    dim3 block_dimension = dim3(CUDA_BLOCK_DIMENSON);
    dim3 grid_dimension = dim3((voxel_count + block_dimension.x - 1) / block_dimension.x);

    image_scale_kernel<<<grid_dimension, block_dimension>>>(
         image1, factor, result, voxel_count);
    cudaCheckError( cudaDeviceSynchronize() );
}


template<typename Pixel>
__global__ void image_assign_kernel(
        Pixel* source, Pixel* destination,
        Dimension voxel_count)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= voxel_count)
        return;

    destination[index] = source[index];
}
template<typename Pixel>
void image_assign_kernel_launch(Pixel* source, Pixel* destination,
                             Dimension voxel_count)
{
    dim3 block_dimension = dim3(CUDA_BLOCK_DIMENSON);
    dim3 grid_dimension = dim3((voxel_count + block_dimension.x - 1) / block_dimension.x);

    image_assign_kernel<<<grid_dimension, block_dimension>>>(
         source, destination, voxel_count);
    cudaCheckError( cudaDeviceSynchronize() );
}



template void image_assign_kernel_launch(float* source, float* destination, Dimension voxel_count);
template void image_scale_kernel_launch(float* image1, float factor, float* result, Dimension voxel_count);
template void image_subtract_kernel_launch(float* image1, float* image2, float* result, Dimension voxel_count);
template void image_add_kernel_launch(float* image1, float* image2, float* result, Dimension voxel_count);
template void image_multiply_matrix_kernel_launch(float* matrix_elements, float* vector, float* result,
                                                    Dimension voxel_count);
template void image_multiply_kernel_launch(float* image1, float* image2, float* temp, Dimension voxel_count);

template void image_assign_kernel_launch(double* source, double* destination, Dimension voxel_count);
template void image_scale_kernel_launch(double* image1, double factor, double* result, Dimension voxel_count);
template void image_subtract_kernel_launch(double* image1, double* image2, double* result, Dimension voxel_count);
template void image_add_kernel_launch(double* image1, double* image2, double* result, Dimension voxel_count);
template void image_multiply_matrix_kernel_launch(double* matrix_elements, double* vector, double* result,
                                                    Dimension voxel_count);
template void image_multiply_kernel_launch(double* image1, double* image2, double* temp, Dimension voxel_count);
