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

template<typename Pixel>
__global__ void image_laplace_kernel(
        Pixel* source, Pixel* destination,
        Dimension voxel_count, Dimension width, Dimension height, Dimension depth)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= voxel_count)
        return;

    const int z_offset = width*height;

    const int z = floorf(index / z_offset);
    int index_rest = index - z * z_offset;
    const int y = floorf(index_rest / width);
    index_rest = index_rest - y * width;
    const int x = index_rest;


    destination[index] = 0;
    Pixel center = 0;

    if(x > 0)
    {
        destination[index] -= source[index - 1];
        center += 1;
    }
    if(x < width - 1)
    {
        destination[index] -= source[index + 1];
        center += 1;
    }

    if(y > 0)
    {
        destination[index] -= source[index - width];
        center += 1;
    }
    if(y < height - 1)
    {
        destination[index] -= source[index + width];
        center += 1;
    }

    if(z > 0)
    {
        destination[index] -= source[index - z_offset];
        center += 1;
    }
    if(z < depth - 1)
    {
        destination[index] -= source[index + z_offset];
        center += 1;
    }

    destination[index] += source[index] * center;
}

template<typename Pixel>
void laplace_kernel_launch(Pixel* source, Pixel* destination,
                           Dimension image_width, Dimension image_height, Dimension image_depth)
{
    Dimension voxel_count = image_width * image_height * image_depth;

    dim3 block_dimension = dim3(CUDA_BLOCK_DIMENSON);
    dim3 grid_dimension = dim3((voxel_count + block_dimension.x - 1) / block_dimension.x);

    image_laplace_kernel<<<grid_dimension, block_dimension>>>(
         source, destination, voxel_count, image_width, image_height, image_depth);
    cudaCheckError( cudaDeviceSynchronize() );
}

template<typename Pixel>
__global__ void image_set_zeros_kernel(
        Pixel* source,
        Dimension voxel_count)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= voxel_count)
        return;

    source[index] = 0;
}

template<typename Pixel>
void image_set_zeros_kernel_launch(Pixel* source,
                           Dimension voxel_count)
{
    dim3 block_dimension = dim3(CUDA_BLOCK_DIMENSON);
    dim3 grid_dimension = dim3((voxel_count + block_dimension.x - 1) / block_dimension.x);

    image_set_zeros_kernel<<<grid_dimension, block_dimension>>>(
         source, voxel_count);
    cudaCheckError( cudaDeviceSynchronize() );
}




template void image_set_zeros_kernel_launch(float* source, Dimension voxel_count);
template void laplace_kernel_launch(float* source, float* destination,
                           Dimension image_width, Dimension image_height, Dimension image_depth);
template void image_assign_kernel_launch(float* source, float* destination, Dimension voxel_count);
template void image_scale_kernel_launch(float* image1, float factor, float* result, Dimension voxel_count);
template void image_subtract_kernel_launch(float* image1, float* image2, float* result, Dimension voxel_count);
template void image_add_kernel_launch(float* image1, float* image2, float* result, Dimension voxel_count);
template void image_multiply_matrix_kernel_launch(float* matrix_elements, float* vector, float* result,
                                                    Dimension voxel_count);
template void image_multiply_kernel_launch(float* image1, float* image2, float* temp, Dimension voxel_count);

template void image_set_zeros_kernel_launch(double* source, Dimension voxel_count);
template void laplace_kernel_launch(double* source, double* destination,
                           Dimension image_width, Dimension image_height, Dimension image_depth);
template void image_assign_kernel_launch(double* source, double* destination, Dimension voxel_count);
template void image_scale_kernel_launch(double* image1, double factor, double* result, Dimension voxel_count);
template void image_subtract_kernel_launch(double* image1, double* image2, double* result, Dimension voxel_count);
template void image_add_kernel_launch(double* image1, double* image2, double* result, Dimension voxel_count);
template void image_multiply_matrix_kernel_launch(double* matrix_elements, double* vector, double* result,
                                                    Dimension voxel_count);
template void image_multiply_kernel_launch(double* image1, double* image2, double* temp, Dimension voxel_count);
