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

template<typename Pixel>
void unary_operation_part1(Pixel* image_host,
                  uint width, uint height, uint depth,
                  Pixel** image,
                  dim3& block_dimension,
                  dim3& grid_dimension)
{
    int cuda_device_count;
    cudaCheckError( cudaGetDeviceCount(&cuda_device_count) );

//    printf("found %d cuda devices.\n", cuda_device_count);

    uint voxel_count = width*height*depth;
    block_dimension = dim3(CUDA_BLOCK_DIMENSON);
    grid_dimension = dim3((voxel_count + block_dimension.x - 1) / block_dimension.x);

    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMalloc(image, size) )
    cudaCheckError( cudaMemcpy(*image, image_host, size, cudaMemcpyHostToDevice) )
    cudaCheckError( cudaDeviceSynchronize() );
}


template<typename Pixel>
Pixel* unary_operation_part2(Pixel* image,
                  uint width, uint height, uint depth)
{
    uint voxel_count = width*height*depth;
    Pixel* result = new Pixel[voxel_count];

    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMemcpy(result, image, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaDeviceSynchronize() );

    cudaCheckError( cudaFree(image) );

    return result;
}
