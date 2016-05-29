

#include "tgv_common.cu"

template<typename Pixel>
__global__ void gradient_magnitude_kernel(
        Pixel* p_x, Pixel* p_y, Pixel* p_z,
        const uint width, const uint height, const uint depth)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    Pixel magnitude =
            p_x[index] * p_x[index] +
            p_y[index] * p_y[index];
    if(depth > 1)
        magnitude += p_z[index] * p_z[index];

    p_x[index] = magnitude;
}

template<typename Pixel>
Pixel* gradient_magnitude_kernel_launch(Pixel* f_host,
                  uint width, uint height, uint depth) {
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

    tgv_launch_forward_differences<Pixel>(f,
            p_x, p_y, p_z,
            width, height, depth,
            block_dimension,
            grid_dimension_x,
            grid_dimension_y,
            grid_dimension_z);

    gradient_magnitude_kernel<<<grid_dimension, block_dimension>>>(
         p_x, p_y, p_z,
         width, height, depth);
    cudaCheckError( cudaDeviceSynchronize() );


    Pixel* result = new Pixel[voxel_count];

    cudaCheckError( cudaMemcpy(result, p_x, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaDeviceSynchronize() );

    cudaCheckError( cudaFree(f) );
    cudaCheckError( cudaFree(p_x) );
    cudaCheckError( cudaFree(p_y) );
    if(depth > 1)
        cudaCheckError( cudaFree(p_z) );

    return result;
}


// THRESHOLD ...

template<typename Pixel>
__global__ void threshold_upper_kernel(
        Pixel* f,
        const uint width, const uint height, const uint depth, const Pixel threshold_value)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    f[index] = f[index] > threshold_value ? 0 : 1;
}

template<typename Pixel>
Pixel* threshold_upper_kernel_launch(Pixel* f_host,
                  uint width, uint height, uint depth,
                  Pixel threshold_value) {
    int cuda_device_count;
    cudaCheckError( cudaGetDeviceCount(&cuda_device_count) );

    printf("found %d cuda devices.\n", cuda_device_count);

    uint voxel_count = width*height*depth;
    dim3 block_dimension(CUDA_BLOCK_DIMENSON);
    dim3 grid_dimension((voxel_count + block_dimension.x - 1) / block_dimension.x);

    Pixel* f;
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMallocManaged(&f, size) )
    cudaCheckError( cudaMemcpy(f, f_host, size, cudaMemcpyHostToDevice) )

    threshold_upper_kernel<<<grid_dimension, block_dimension>>>(
         f, width, height, depth, threshold_value);

    Pixel* result = new Pixel[voxel_count];

    cudaCheckError( cudaMemcpy(result, f, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaDeviceSynchronize() );

    cudaCheckError( cudaFree(f) );

    return result;
}

// multiply ...

template<typename Pixel>
__global__ void multiply_kernel(
        Pixel* image1, Pixel* image2,
        const uint width, const uint height, const uint depth)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    image1[index] = image1[index] * image2[index];
}

template<typename Pixel>
Pixel* multiply_kernel_launch(Pixel* image1_host, Pixel* image2_host,
                  uint width, uint height, uint depth)
{
    int cuda_device_count;
    cudaCheckError( cudaGetDeviceCount(&cuda_device_count) );

    printf("found %d cuda devices.\n", cuda_device_count);

    uint voxel_count = width*height*depth;
    dim3 block_dimension(CUDA_BLOCK_DIMENSON);
    dim3 grid_dimension((voxel_count + block_dimension.x - 1) / block_dimension.x);

    Pixel* image1, *image2;
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMallocManaged(&image1, size) )
    cudaCheckError( cudaMemcpy(image1, image1_host, size, cudaMemcpyHostToDevice) )
    cudaCheckError( cudaMallocManaged(&image2, size) )
    cudaCheckError( cudaMemcpy(image2, image2_host, size, cudaMemcpyHostToDevice) )

    multiply_kernel<<<grid_dimension, block_dimension>>>(
         image1, image2, width, height, depth);

    Pixel* result = new Pixel[voxel_count];

    cudaCheckError( cudaMemcpy(result, image1, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaDeviceSynchronize() );

    cudaCheckError( cudaFree(image1) );
    cudaCheckError( cudaFree(image2) );

    return result;
}

// gradient x, y, z

template<typename Pixel>
void gradient_kernel_launch(Pixel* f_host,
                              uint width, uint height, uint depth,
                              Pixel** gradient_x_host,
                              Pixel** gradient_y_host,
                              Pixel** gradient_z_host)
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

    tgv_launch_forward_differences<Pixel>(f,
            p_x, p_y, p_z,
            width, height, depth,
            block_dimension,
            grid_dimension_x,
            grid_dimension_y,
            grid_dimension_z);

    *gradient_x_host = new Pixel[voxel_count];
    *gradient_y_host = new Pixel[voxel_count];
    if(depth > 1)
        *gradient_z_host = new Pixel[voxel_count];

    cudaCheckError( cudaMemcpy(*gradient_x_host, p_x, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaMemcpy(*gradient_y_host, p_y, size, cudaMemcpyDeviceToHost) );

    if(depth > 1)
        cudaCheckError( cudaMemcpy(*gradient_z_host, p_z, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaDeviceSynchronize() );

    cudaCheckError( cudaFree(f) );
    cudaCheckError( cudaFree(p_x) );
    cudaCheckError( cudaFree(p_y) );

    if(depth > 1)
        cudaCheckError( cudaFree(p_z) );
}

template float* gradient_magnitude_kernel_launch(float* f_host,
      uint width, uint height, uint depth);
template double* gradient_magnitude_kernel_launch(double* f_host,
      uint width, uint height, uint depth);

template float* threshold_upper_kernel_launch(float* f_host,
                  uint width, uint height, uint depth,
                  float threshold_value);
template double* threshold_upper_kernel_launch(double* f_host,
                  uint width, uint height, uint depth,
                  double threshold_value);

template float* multiply_kernel_launch(float* image1, float* image2,
                  uint width, uint height, uint depth);
template double* multiply_kernel_launch(double* image1, double* image2,
                  uint width, uint height, uint depth);

template void gradient_kernel_launch(float* f_host,
                              uint width, uint height, uint depth,
                              float** gradient_x_host,
                              float** gradient_y_host,
                              float** gradient_z_host);
template void gradient_kernel_launch(double* f_host,
                              uint width, uint height, uint depth,
                              double** gradient_x_host,
                              double** gradient_y_host,
                              double** gradient_z_host);
