#ifndef TGV_3_COMMON
#define TGV_3_COMMON

#include "tgv2_common.cu"

template<typename Pixel>
__global__ void tgv3_kernel_part5(
        Pixel* v_x,Pixel* v_y,Pixel* v_z,
        Pixel* v_previous_x, Pixel* v_previous_y, Pixel* v_previous_z,
        Pixel* q_x,Pixel* q_y,Pixel* q_z,
        Pixel* q_xy,Pixel* q_xz,Pixel* q_yz,
        Pixel* q_x2, Pixel* q_y2, Pixel* q_z2,
        Pixel* q_xy2, Pixel* q_xz2, Pixel* q_yz2,

        Pixel* w_x,Pixel* w_y,Pixel* w_z,
        Pixel* w_xy,Pixel* w_xz,Pixel* w_yz,

        const Pixel sigma, const Pixel alpha0,
        const uint width, const uint height, const uint depth) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

       /*
        * Matlab Code:
        % dual update q
        q = q + sigma*nabla_second*v_bar;
        norm_q = sqrt(q(1:N).^2 + q(N+1:2*N).^2 + q(2*N+1:3*N).^2 + ... % main diagonal
            2*q(3*N+1:4*N).^2 + 2*q(4*N+1:5*N).^2 + 2*q(5*N+1:6*N).^2); % off diagonal
        q = q./max(1, repmat(norm_q, 6, 1)/alpha0);
       */

    q_x2[index] += sigma * (q_x[index] - w_x[index]);
    q_y2[index] += sigma * (q_y[index] - w_y[index]);
    q_xy2[index] += sigma * (q_xy[index] - w_xy[index]);

    if(depth > 1) {
        q_z2[index] += sigma * (q_z[index] - w_z[index]);
        q_xz2[index] += sigma * (q_xz[index] - w_xz[index]);
        q_yz2[index] += sigma * (q_yz[index] - w_yz[index]);
    }

    Pixel normalization =
            q_x2[index] * q_x2[index] +
            q_y2[index] * q_y2[index] +
            2 * q_xy2[index] * q_xy2[index];

    if(depth > 1)
        normalization += q_z2[index] * q_z2[index] +
                2 * q_xz2[index] * q_xz2[index] +
                2 * q_yz2[index] * q_yz2[index];

    normalization = fmax(1, sqrtf(normalization) / alpha0);

    q_x2[index] /= normalization;
    q_y2[index] /= normalization;
    q_xy2[index] /= normalization;
    if(depth > 1) {
        q_z2[index] /= normalization;
        q_xz2[index] /= normalization;
        q_yz2[index] /= normalization;

        v_previous_z[index] = v_z[index];
    }

    v_previous_x[index] = v_x[index];
    v_previous_y[index] = v_y[index];
}


template<typename Pixel>
void tgv3_launch_part23(
        uint voxel_count, uint depth,

        Pixel** w_x, Pixel** w_y, Pixel** w_z,
        Pixel** w_xy, Pixel** w_xz, Pixel** w_yz,

        Pixel** w_bar_x, Pixel** w_bar_y, Pixel** w_bar_z,
        Pixel** w_bar_xy, Pixel** w_bar_xz, Pixel** w_bar_yz,

        Pixel** w_previous_x, Pixel** w_previous_y, Pixel** w_previous_z,
        Pixel** w_previous_xy, Pixel** w_previous_xz, Pixel** w_previous_yz,

        Pixel** r_x, Pixel** r_y, Pixel** r_z,
        Pixel** r_xy, Pixel** r_xz, Pixel** r_yz,

        Pixel** r2_x, Pixel** r2_y, Pixel** r2_z,
        Pixel** r2_xy, Pixel** r2_xz, Pixel** r2_yz)
{
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMallocManaged(w_x, size) )
    cudaCheckError( cudaMallocManaged(w_bar_x, size) )
    cudaCheckError( cudaMallocManaged(w_previous_x, size) )

    cudaCheckError( cudaMallocManaged(w_y, size) )
    cudaCheckError( cudaMallocManaged(w_bar_y, size) )
    cudaCheckError( cudaMallocManaged(w_previous_y, size) )

    cudaCheckError( cudaMallocManaged(w_xy, size) )
    cudaCheckError( cudaMallocManaged(w_bar_xy, size) )
    cudaCheckError( cudaMallocManaged(w_previous_xy, size) )

    cudaCheckError( cudaMallocManaged(r_x, size) )
    cudaCheckError( cudaMallocManaged(r_y, size) )
    cudaCheckError( cudaMallocManaged(r_xy, size) )

    cudaCheckError( cudaMallocManaged(r2_x, size) )
    cudaCheckError( cudaMallocManaged(r2_y, size) )
    cudaCheckError( cudaMallocManaged(r2_xy, size) )

    if(depth > 1) {
        cudaCheckError( cudaMallocManaged(w_z, size) )
        cudaCheckError( cudaMallocManaged(w_bar_z, size) )
        cudaCheckError( cudaMallocManaged(w_previous_z, size) )

        cudaCheckError( cudaMallocManaged(w_xz, size) )
        cudaCheckError( cudaMallocManaged(w_bar_xz, size) )
        cudaCheckError( cudaMallocManaged(w_previous_xz, size) )

        cudaCheckError( cudaMallocManaged(w_yz, size) )
        cudaCheckError( cudaMallocManaged(w_bar_yz, size) )
        cudaCheckError( cudaMallocManaged(w_previous_yz, size) )

        cudaCheckError( cudaMallocManaged(r_z, size) )
        cudaCheckError( cudaMallocManaged(r_xz, size) )
        cudaCheckError( cudaMallocManaged(r_yz, size) )

        cudaCheckError( cudaMallocManaged(r2_z, size) )
        cudaCheckError( cudaMallocManaged(r2_xz, size) )
        cudaCheckError( cudaMallocManaged(r2_yz, size) )
    }
}


template<typename Pixel>
void tgv3_launch_part33(
        uint depth,

        Pixel* w_x, Pixel* w_y, Pixel* w_z,
        Pixel* w_xy, Pixel* w_xz, Pixel* w_yz,

        Pixel* w_bar_x, Pixel* w_bar_y, Pixel* w_bar_z,
        Pixel* w_bar_xy, Pixel* w_bar_xz, Pixel* w_bar_yz,

        Pixel* w_previous_x, Pixel* w_previous_y, Pixel* w_previous_z,
        Pixel* w_previous_xy, Pixel* w_previous_xz, Pixel* w_previous_yz,

        Pixel* r_x, Pixel* r_y, Pixel* r_z,
        Pixel* r_xy, Pixel* r_xz, Pixel* r_yz,

        Pixel* r2_x, Pixel* r2_y, Pixel* r2_z,
        Pixel* r2_xy, Pixel* r2_xz, Pixel* r2_yz)
{
    cudaFree(w_x);
    cudaFree(w_y);
    cudaFree(w_xy);

    cudaFree(w_bar_x);
    cudaFree(w_bar_y);
    cudaFree(w_bar_xy);

    cudaFree(w_previous_x);
    cudaFree(w_previous_y);
    cudaFree(w_previous_xy);

    cudaFree(r_x);
    cudaFree(r_y);
    cudaFree(r_xy);

    cudaFree(r2_x);
    cudaFree(r2_y);
    cudaFree(r2_xy);

    if(depth > 1) {
        cudaFree(w_z);
        cudaFree(w_xz);
        cudaFree(w_yz);

        cudaFree(w_bar_z);
        cudaFree(w_bar_xz);
        cudaFree(w_bar_yz);

        cudaFree(w_previous_z);
        cudaFree(w_previous_xz);
        cudaFree(w_previous_yz);

        cudaFree(r_z);
        cudaFree(r_xz);
        cudaFree(r_yz);

        cudaFree(r2_z);
        cudaFree(r2_xz);
        cudaFree(r2_yz);
    }
}

template<typename Pixel>
void tgv_launch_gradient3(
        Pixel* w_bar_x, Pixel* w_bar_y, Pixel* w_bar_z,
        Pixel* w_bar_xy, Pixel* w_bar_xz, Pixel* w_bar_yz,

        Pixel* r_x, Pixel* r_y, Pixel* r_z,
        Pixel* r_xy, Pixel* r_xz, Pixel* r_yz,  Pixel* q_temp,
        uint width, uint height, uint depth,
        dim3 block_dimension,
        dim3 grid_dimension,
        dim3 grid_dimension_x,
        dim3 grid_dimension_y,
        dim3 grid_dimension_z)
{
    forward_difference_x<<<grid_dimension_x, block_dimension>>>(
          w_bar_x, r_x, width, height, depth);

    forward_difference_y<<<grid_dimension_y, block_dimension>>>(
          w_bar_y, r_y, width, height, depth);

    forward_difference_x<<<grid_dimension_x, block_dimension>>>(
          w_bar_xy, r_xy, width, height, depth);
    forward_difference_y<<<grid_dimension_y, block_dimension>>>(
          w_bar_xy, q_temp, width, height, depth);
    addAndHalf<<<grid_dimension, block_dimension>>>(
            r_xy, q_temp, r_xy,
            width, height, depth);

    if(depth > 1) {
        forward_difference_z<<<grid_dimension_z, block_dimension>>>(
              w_bar_z, r_z, width, height, depth);

        forward_difference_x<<<grid_dimension_x, block_dimension>>>(
              w_bar_xz, r_xz, width, height, depth);
        forward_difference_z<<<grid_dimension_z, block_dimension>>>(
              w_bar_xz, q_temp, width, height, depth);
        addAndHalf<<<grid_dimension, block_dimension>>>(
                r_xz, q_temp, r_xz,
                width, height, depth);

        forward_difference_y<<<grid_dimension_y, block_dimension>>>(
              w_bar_yz, r_yz, width, height, depth);
        forward_difference_z<<<grid_dimension_z, block_dimension>>>(
              w_bar_yz, q_temp, width, height, depth);
        addAndHalf<<<grid_dimension, block_dimension>>>(
                r_yz, q_temp, r_yz,
                width, height, depth);
    }
    cudaCheckError( cudaDeviceSynchronize() );
}


template<typename Pixel>
void tgv_launch_divergence3(
        Pixel* r_x, Pixel* r_y, Pixel* r_z,
        Pixel* r_xy, Pixel* r_xz, Pixel* r_yz,

        Pixel* r2_x, Pixel* r2_y, Pixel* r2_z,
        Pixel* r2_xy, Pixel* r2_xz, Pixel* r2_yz,

        Pixel* r_temp,

        uint width, uint height, uint depth,
        dim3 block_dimension,
        dim3 grid_dimension,
        dim3 grid_dimension_x,
        dim3 grid_dimension_y,
        dim3 grid_dimension_z)
{
    backward_difference_x<<<grid_dimension_x, block_dimension>>>(
      r_x, r2_x, width, height, depth);

    backward_difference_y<<<grid_dimension_y, block_dimension>>>(
      r_y, r2_y, width, height, depth);

    backward_difference_x<<<grid_dimension_x, block_dimension>>>(
      r_xy, r2_xy, width, height, depth);
    backward_difference_y<<<grid_dimension_y, block_dimension>>>(
      r_xy, r_temp, width, height, depth);
    add<<<grid_dimension, block_dimension>>>(r2_xy, r_temp, r2_xy, width, height, depth);

    if(depth > 1) {
        backward_difference_z<<<grid_dimension_z, block_dimension>>>(
          r_z, r2_z, width, height, depth);

        backward_difference_x<<<grid_dimension_x, block_dimension>>>(
          r_xz, r2_xz, width, height, depth);
        backward_difference_z<<<grid_dimension_z, block_dimension>>>(
          r_xz, r_temp, width, height, depth);
        add<<<grid_dimension, block_dimension>>>(r2_xz, r_temp, r2_xz, width, height, depth);

        backward_difference_y<<<grid_dimension_y, block_dimension>>>(
          r_yz, r2_yz, width, height, depth);
        backward_difference_z<<<grid_dimension_z, block_dimension>>>(
          r_yz, r_temp, width, height, depth);
        add<<<grid_dimension, block_dimension>>>(r2_yz, r_temp, r2_yz, width, height, depth);
    }
    cudaCheckError( cudaDeviceSynchronize() );
}

template<typename Pixel>
__global__ void tgv3_kernel_part52(
        Pixel* r2_x, Pixel* r2_y, Pixel* r2_z,
        Pixel* r2_xy, Pixel* r2_xz, Pixel* r2_yz,

        Pixel* r_x, Pixel* r_y, Pixel* r_z,
        Pixel* r_xy, Pixel* r_xz, Pixel* r_yz,

        Pixel* w_x, Pixel* w_y, Pixel* w_z,
        Pixel* w_xy, Pixel* w_xz, Pixel* w_yz,

        Pixel* w_previous_x, Pixel* w_previous_y, Pixel* w_previous_z,
        Pixel* w_previous_xy, Pixel* w_previous_xz, Pixel* w_previous_yz,

        const Pixel sigma, const Pixel alpha2,
        const uint width, const uint height, const uint depth) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    r_x[index] += sigma * r2_x[index];
    r_y[index] += sigma * r2_y[index];
    r_xy[index] += sigma * r2_xy[index];

    if(depth > 1) {
        r_z[index] += sigma * r2_z[index];
        r_xz[index] += sigma * r2_xz[index];
        r_yz[index] += sigma * r2_yz[index];
    }

    Pixel normalization =
            r_x[index] * r_x[index] +
            r_y[index] * r_y[index] +
            2 * r_xy[index] * r_xy[index];

    if(depth > 1)
        normalization += r_z[index] * r_z[index] +
                2 * r_xz[index] * r_xz[index] +
                2 * r_yz[index] * r_yz[index];

    normalization = fmax(1, sqrtf(normalization) / alpha2);

    r_x[index] /= normalization;
    r_y[index] /= normalization;
    r_xy[index] /= normalization;
    if(depth > 1) {
        r_z[index] /= normalization;
        r_xz[index] /= normalization;
        r_yz[index] /= normalization;

        w_previous_z[index] = w_z[index];
        w_previous_xz[index] = w_xz[index];
        w_previous_yz[index] = w_yz[index];
    }

    w_previous_x[index] = w_x[index];
    w_previous_y[index] = w_y[index];
    w_previous_xy[index] = w_xy[index];
}

template<typename Pixel>
__global__ void tgv3_kernel_part62(
        Pixel* r2_x, Pixel* r2_y, Pixel* r2_z,
        Pixel* r2_xy, Pixel* r2_xz, Pixel* r2_yz,

        Pixel* q_x, Pixel* q_y, Pixel* q_z,
        Pixel* q_xy, Pixel* q_xz, Pixel* q_yz,

        Pixel* w_x, Pixel* w_y, Pixel* w_z,
        Pixel* w_xy, Pixel* w_xz, Pixel* w_yz,

        Pixel* w_previous_x, Pixel* w_previous_y, Pixel* w_previous_z,
        Pixel* w_previous_xy, Pixel* w_previous_xz, Pixel* w_previous_yz,

        Pixel* w_bar_x, Pixel* w_bar_y, Pixel* w_bar_z,
        Pixel* w_bar_xy, Pixel* w_bar_xz, Pixel* w_bar_yz,

        const Pixel tau, const Pixel theta,
        const uint width, const uint height, const uint depth)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    w_x[index] -= tau * (r2_x[index] - q_x[index]);
    w_y[index] -= tau * (r2_y[index] - q_y[index]);

    w_xy[index] -= tau * (r2_xy[index] - q_xy[index]);

    if(depth > 1) {
        w_z[index] -= tau * (r2_z[index] - q_z[index]);

        w_xz[index] -= tau * (r2_xz[index] - q_xz[index]);
        w_yz[index] -= tau * (r2_yz[index] - q_yz[index]);

        w_bar_z[index] = w_z[index] + theta*(w_z[index] - w_previous_z[index]);
        w_bar_xz[index] = w_xz[index] + theta*(w_xz[index] - w_previous_xz[index]);
        w_bar_yz[index] = w_yz[index] + theta*(w_yz[index] - w_previous_yz[index]);
    }


    w_bar_x[index] = w_x[index] + theta*(w_x[index] - w_previous_x[index]);
    w_bar_y[index] = w_y[index] + theta*(w_y[index] - w_previous_y[index]);
    w_bar_xy[index] = w_xy[index] + theta*(w_xy[index] - w_previous_xy[index]);
}

#endif // TGV_3_COMMON
