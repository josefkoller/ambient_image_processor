#include "tgv_common.cu"
#include "add.cu"

template<typename Pixel>
void launch_divergence(
        Pixel* dx, Pixel* dy, Pixel* dz,
        Pixel* dxdx, Pixel* dydy, Pixel* dzdz,

        const uint width, const uint height, const uint depth,

        dim3 block_dimension,
        dim3 grid_dimension,
        dim3 grid_dimension_x,
        dim3 grid_dimension_y,
        dim3 grid_dimension_z)
{
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
    cudaCheckError( cudaDeviceSynchronize() );
    if(depth > 1)
    {
        add_kernel<<<grid_dimension, block_dimension>>>(
             dxdx, dzdz, width, height, depth);
        cudaCheckError( cudaDeviceSynchronize() );
    }
}

template<typename Pixel>
Pixel* divergence_kernel_launch(
        Pixel* dx, Pixel* dy, Pixel* dz,
        const uint width, const uint height, const uint depth, bool is_host_data=false)
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

    if(is_host_data)
    {
        uint voxel_count = width*height*depth;
        size_t size = sizeof(Pixel) * voxel_count;

        Pixel *dx2, *dy2, *dz2;
        cudaCheckError( cudaMallocManaged(&dx2, size) )
        cudaCheckError( cudaMallocManaged(&dy2, size) )
        if(depth > 1)
          cudaCheckError( cudaMallocManaged(&dz2, size) )
        cudaCheckError( cudaDeviceSynchronize() );

        cudaCheckError( cudaMemcpy(dx2, dx, size, cudaMemcpyHostToDevice) )
        dx = dx2;
        cudaCheckError( cudaMemcpy(dy2, dy, size, cudaMemcpyHostToDevice) )
        dy = dy2;
        if(depth > 1)
        {
            cudaCheckError( cudaMemcpy(dz2, dz, size, cudaMemcpyHostToDevice) )
            dz = dz2;
        }
        cudaCheckError( cudaDeviceSynchronize() );
    }

    Pixel *dxdx, *dydy, *dzdz;
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMallocManaged(&dxdx, size) )
    cudaCheckError( cudaMallocManaged(&dydy, size) )
    if(depth > 1)
      cudaCheckError( cudaMallocManaged(&dzdz, size) )
    cudaCheckError( cudaDeviceSynchronize() );

    launch_divergence(dx, dy, dz,
                      dxdx, dydy, dzdz,
                      width, height, depth,
                      block_dimension,
                      grid_dimension,
                      grid_dimension_x,
                      grid_dimension_y,
                      grid_dimension_z);
    if(depth > 1)
    {
        cudaFree(dzdz);
    }

    Pixel* result = new Pixel[voxel_count];
    cudaCheckError( cudaMemcpy(result, dxdx, size, cudaMemcpyDeviceToHost) )
    cudaCheckError( cudaDeviceSynchronize() );

    cudaFree(dxdx);
    cudaFree(dydy);
    cudaCheckError( cudaDeviceSynchronize() );

    if(is_host_data)
    {
        cudaFree(dx);
        cudaFree(dy);
        if(depth > 1)
            cudaFree(dz);
    }

    return result;
}

template float* divergence_kernel_launch(
float* dx, float* dy, float* dz,
const uint width, const uint height, const uint depth, bool is_host_data);

template double* divergence_kernel_launch(
double* dx, double* dy, double* dz,
const uint width, const uint height, const uint depth, bool is_host_data);
