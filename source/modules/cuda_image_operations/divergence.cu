#include "tgv_common.cu"
#include "add.cu"

template<typename Pixel>
__global__ void divergence_kernel(
        Pixel* image1, Pixel* image2,
        const uint width, const uint height, const uint depth)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    image1[index] = image1[index] * image2[index];
}

template<typename Pixel>
Pixel* divergence_kernel_launch(
        Pixel* dx, Pixel* dy, Pixel* dz,
        const uint width, const uint height, const uint depth)
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

    Pixel *dxdx, *dydy, *dzdz;
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMallocManaged(&dxdx, size) )
    cudaCheckError( cudaMallocManaged(&dydy, size) )
    if(depth > 1)
      cudaCheckError( cudaMallocManaged(&dzdz, size) )

    tgv_launch_backward_differences<Pixel>(
            dxdx, dydy, dzdz,
            dx, dy, dz,
            width, height, depth,
            block_dimension,
            grid_dimension_x,
            grid_dimension_y,
            grid_dimension_z);

    add_kernel<<<grid_dimension, block_dimension>>>(
         dxdx, dydy, width, height, depth);
    if(depth > 1)
    {
        add_kernel<<<grid_dimension, block_dimension>>>(
             dxdx, dzdz, width, height, depth);
        cudaFree(dzdz);
    }

    Pixel* result = new Pixel[voxel_count];
    cudaCheckError( cudaMemcpy(result, dxdx, size, cudaMemcpyDeviceToHost) )

    cudaFree(dxdx);
    cudaFree(dydy);

    return result;
}

template float* divergence_kernel_launch(
float* dx, float* dy, float* dz,
const uint width, const uint height, const uint depth);

template double* divergence_kernel_launch(
double* dx, double* dy, double* dz,
const uint width, const uint height, const uint depth);
