
#include "tgv_common.cu"

template<typename Pixel>
__global__ void tgv_kernel_part4_l2(
        Pixel* p_x, Pixel* p_y, Pixel* p_z,
        Pixel* p_xx, Pixel* p_yy, Pixel* p_zz,
        const Pixel tau, Pixel* u, Pixel* f,
        const Pixel lambda,
        Pixel* u_previous, const Pixel theta, Pixel* u_bar,
        const uint width, const uint height, const uint depth)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    if(depth > 1)
        u[index] -= tau * (p_x[index] + p_y[index] + p_z[index]);
    else
        u[index] -= tau * (p_x[index] + p_y[index]);

    u[index] = (u[index] + tau*lambda*f[index]) / (1 + tau*lambda);

    u_bar[index] = u[index] + theta*(u[index] - u_previous[index]);

    /*
     *  Matlab Code:
          u = u - tau * nabla_t * p;
          u = (u + tau * lambda .* f)/(1 + tau * lambda);

          % overrelaxation
          u_bar = u + theta*(u - u_old);
    */
}


template<typename Pixel>
Pixel* tgv1_l2_launch(Pixel* f_host,
                  uint width, uint height, uint depth,
                  Pixel lambda,
                  uint iteration_count,
                  uint paint_iteration_interval,
                  IterationCallback<Pixel> iteration_finished_callback,
                  Pixel alpha0,
                  Pixel alpha1)
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

    Pixel* f, *u;
    Pixel* u_previous, *u_bar, *p_x, *p_y, *p_z, *p_xx, *p_yy, *p_zz;

    tgv_launch_part2<Pixel>(f_host,
                voxel_count, depth,
                &f, &u,
                &u_previous, &u_bar, &p_x, &p_y, &p_z, &p_xx, &p_yy, &p_zz);

    // algorithm variables..
    const Pixel sqrt_8 = std::sqrt(8.0);
    const Pixel tau = 1.0 / sqrt_8;
    const Pixel sigma = tau;
    const Pixel theta = 1;

    // algorithm begin
    zeroInit<<<grid_dimension, block_dimension>>>(
                                                    p_x, p_y, p_z,
                                                    p_xx, p_yy, p_zz,
                                                    voxel_count, depth);
    clone2<<<grid_dimension, block_dimension>>>(
                                                  f, u, u_bar, voxel_count);
    cudaCheckError( cudaDeviceSynchronize() );

    for(uint iteration_index = 0; iteration_index < iteration_count; iteration_index++)
    {
        /* matlab primal dual TVL2
          % dual update
          p = p + sigma*nabla*u_bar;
          norm_p = sqrt(p(1:N).^2 + p(N+1:2*N).^2);
          p = p./max(1,[norm_p; norm_p]);

          u_old = u;

          % primal update
          u = u - tau * nabla_t * p;
          u = (u + tau * lambda .* f)/(1 + tau * lambda);

          % overrelaxation
          u_bar = u + theta*(u - u_old);
      */

        tgv_launch_forward_differences<Pixel>(u_bar,
                p_x,p_y,p_z,
                width, height, depth,
                block_dimension,
                grid_dimension_x,
                grid_dimension_y,
                grid_dimension_z);

        tgv_kernel_part2<<<grid_dimension, block_dimension>>>(
                                                                p_x, p_y, p_z,
                                                                p_xx, p_yy, p_zz,
                                                                sigma, alpha1, u_previous, u,
                                                                width, height, depth);
        cudaCheckError( cudaDeviceSynchronize() );

        tgv_launch_backward_differences<Pixel>(
                p_x, p_y, p_z,
                p_xx, p_yy, p_zz,
                width, height, depth,
                block_dimension,
                grid_dimension_x,
                grid_dimension_y,
                grid_dimension_z);

        tgv_kernel_part4_l2<<<grid_dimension, block_dimension>>>(  p_x, p_y, p_z,
                                                                p_xx, p_yy, p_zz,
                                                                tau, u, f,
                                                                lambda,
                                                                u_previous, theta, u_bar,
                                                                width, height, depth);
        cudaCheckError( cudaDeviceSynchronize() );

        printf("TVL2, iteration=%d / %d \n", iteration_index, iteration_count);
        if(paint_iteration_interval > 0 &&
                iteration_index % paint_iteration_interval == 0) {
            iteration_finished_callback(iteration_index, iteration_count, u);
        }
    }

    Pixel* destination = new Pixel[voxel_count];
    tgv_launch_part3<Pixel>(
                destination,
                voxel_count, depth,
                u_previous, u_bar,
                p_x, p_y, p_z,
                p_xx, p_yy, p_zz,
                f, u);

    return destination;
}

// generate the algorithm explicitly for...

template float* tgv1_l2_launch(float* f_host,
uint width, uint height, uint depth,
float lambda,
uint iteration_count,
uint paint_iteration_interval,
IterationCallback<float> iteration_finished_callback,
float alpha0,
float alpha1);

template double* tgv1_l2_launch(double* f_host,
uint width, uint height, uint depth,
double lambda,
uint iteration_count,
uint paint_iteration_interval,
IterationCallback<double> iteration_finished_callback,
double alpha0,
double alpha1);
