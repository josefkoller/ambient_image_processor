
#include "tgv_common.cu"

template<typename Pixel>
__global__ void total_variation(
        Pixel* p_x, Pixel* p_y, Pixel* p_z, const uint width, const uint height, const uint depth) {

    const uint index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width*height*depth)
        return;

    p_x[index] = abs(p_x[index]) + abs(p_y[index]);

    if(depth > 1)
        p_x[index] += abs(p_z[index]);
}


template<typename Pixel>
double tv_kernel_launch(Pixel* f_host,
                        uint width, uint height, uint depth)
{
    uint voxel_count;
    dim3 block_dimension;
    dim3 grid_dimension;
    dim3 grid_dimension_x;
    dim3 grid_dimension_y;
    dim3 grid_dimension_z;

    tgv_launch_part1<Pixel>(
                        width, height, depth,
                        voxel_count,
                        block_dimension,
                        grid_dimension,
                        grid_dimension_x,
                        grid_dimension_y,
                        grid_dimension_z);

    Pixel *f, *p_x, *p_y, *p_z;
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMallocManaged(&f, size) )
    cudaCheckError( cudaMemcpy(f, f_host, size, cudaMemcpyHostToDevice) )

    cudaCheckError( cudaMallocManaged(&p_x, size) )
    cudaCheckError( cudaMallocManaged(&p_y, size) )
    if(depth > 1)
        cudaCheckError( cudaMallocManaged(&p_z, size) )

    cudaCheckError( cudaDeviceSynchronize() );

    tgv_launch_forward_differences<Pixel>(f,
            p_x, p_y, p_z,
            width, height, depth,
            block_dimension,
            grid_dimension_x,
            grid_dimension_y,
            grid_dimension_z);
    cudaCheckError( cudaDeviceSynchronize() );

    total_variation<<<grid_dimension, block_dimension>>>(
          p_x, p_y, p_z, width, height, depth);
    cudaCheckError( cudaDeviceSynchronize() );

    cudaCheckError( cudaMemcpy(f_host, p_x, size, cudaMemcpyDeviceToHost) )

    cudaFree(f);
    cudaFree(p_x);
    cudaFree(p_y);
    if(depth > 1)
        cudaFree(p_z);

    double total_variation = 0;
    for(int i = 0; i < voxel_count; i++)
        total_variation += f_host[i];
    return total_variation;
}

template double tv_kernel_launch(float* image_host,
                              uint width, uint height, uint depth);
template double tv_kernel_launch(double* image_host,
                              uint width, uint height, uint depth);
