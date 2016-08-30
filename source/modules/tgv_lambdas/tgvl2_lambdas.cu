

#include "tgv_common.cu"
#include "tgv2_common.cu"
#include "tgv2_l1.cu"

template<typename Pixel>
__global__ void tgv_lambdas_kernel_part4(
        Pixel* p_x, Pixel* p_y, Pixel* p_z,
        Pixel* lambdas,
        Pixel lambda_offset,
        Pixel lambda_factor,
        const Pixel tau, Pixel* u, Pixel* f,
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

    const Pixel tau_x_lambda = tau*(lambdas[index]*lambda_factor + lambda_offset);
    u[index] = (u[index] + tau_x_lambda*f[index]) / (1 + tau_x_lambda);

    u_bar[index] = u[index] + theta*(u[index] - u_previous[index]);

    /*
     *  Matlab Code:
          u = u - tau * nabla_t * p;
          u = (u + tau * lambda .* f)/(1 + tau * lambda);

          % overrelaxation
          u_bar = u + theta*(u - u_old);
    */
}

/*
 *   min { |grad(u) - grad(f)| + }
 *
 * */

template<typename Pixel>
Pixel* tgv2_l1_lambdas_launch(Pixel* f_host,
                              Pixel* lambdas_host,
                              uint width, uint height, uint depth,
                              Pixel lambda_offset,
                              Pixel lambda_factor,
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
                            voxel_count,
                            &f, &u,
                            &u_previous, &u_bar, &p_x, &p_y, &p_z, &p_xx, &p_yy, &p_zz);

    Pixel *v_x, *v_y, *v_z;
    Pixel *v_bar_x, *v_bar_y, *v_bar_z;
    Pixel *v_previous_x, *v_previous_y, *v_previous_z;
    Pixel *q_x, *q_y, *q_z, *q_xy, *q_xz, *q_yz;
    Pixel *q_x2, *q_y2, *q_z2, *q_xy2, *q_xz2, *q_yz2;
    Pixel * q_temp;

    tgv_launch_part22<Pixel>(
                voxel_count,
                &v_previous_x, &v_previous_y, &v_previous_z,
                &v_bar_x, &v_bar_y, &v_bar_z,
                &v_x, &v_y, &v_z,
                &q_x, &q_y, &q_z,
                &q_xy, &q_xz, &q_yz,
                &q_x2, &q_y2, &q_z2,
                &q_xy2, &q_xz2, &q_yz2, &q_temp);

    Pixel* lambdas;
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMallocManaged(&lambdas, size) )
    cudaCheckError( cudaMemcpy(lambdas, lambdas_host, size, cudaMemcpyHostToDevice) )

            // algorithm variables..
            const Pixel tau = 1.0 / std::sqrt(20.0);
    const Pixel sigma = tau;
    const Pixel theta = 1;

    // algorithm begin
    zeroInit<<<grid_dimension, block_dimension>>>(
                                                    p_x, p_y, p_z,
                                                    p_xx, p_yy, p_zz,
                                                    voxel_count);
    zeroInit<<<grid_dimension, block_dimension>>>(
                                                    v_x, v_y, v_z,
                                                    v_bar_x, v_bar_y, v_bar_z,
                                                    voxel_count);
    zeroInit2<<<grid_dimension, block_dimension>>>(
                                                     q_x, q_y, q_z,
                                                     q_xy, q_xz, q_yz,
                                                     voxel_count);
    clone2<<<grid_dimension, block_dimension>>>(
                                                  f, u, u_bar, voxel_count);
    cudaCheckError( cudaDeviceSynchronize() );

    for(uint iteration_index = 0; iteration_index < iteration_count; iteration_index++)
    {
        tgv_launch_forward_differences<Pixel>(u_bar,
                                              p_x,p_y,p_z,
                                              width, height, depth,
                                              block_dimension,
                                              grid_dimension_x,
                                              grid_dimension_y,
                                              grid_dimension_z);

        tgv_kernel_part22<<<grid_dimension, block_dimension>>>(v_bar_x, v_bar_y, v_bar_z,
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
        cudaCheckError( cudaDeviceSynchronize() );

        tgv_lambdas_kernel_part4<<<grid_dimension, block_dimension>>>(p_x, p_y, p_z,
                                                                      lambdas,
                                                                      lambda_offset,
                                                                      lambda_factor,
                                                                      tau, u, f,
                                                                      u_previous, theta, u_bar,
                                                                      width, height, depth);
        cudaCheckError( cudaDeviceSynchronize() );

        /*
        % dual update q
        q = q + sigma*nabla_second*v_bar;
        norm_q = sqrt(q(1:N).^2 + q(N+1:2*N).^2 + q(2*N+1:3*N).^2 + ... % main diagonal
            2*q(3*N+1:4*N).^2 + 2*q(4*N+1:5*N).^2 + 2*q(5*N+1:6*N).^2); % off diagonal
        q = q./max(1, repmat(norm_q, 6, 1)/alpha0);

            v_old = v;
        */
        tgv_launch_gradient2<Pixel>(
                    v_bar_x, v_bar_y, v_bar_z,
                    q_x2,q_y2,q_z2,
                    q_xy2,q_xz2,q_yz2, q_temp,
                    width, height, depth,
                    block_dimension,
                    grid_dimension,
                    grid_dimension_x,
                    grid_dimension_y,
                    grid_dimension_z);

        tgv_kernel_part5<<<grid_dimension, block_dimension>>>(
                                                                v_x, v_y, v_z,
                                                                v_previous_x, v_previous_y, v_previous_z,
                                                                q_x2,q_y2,q_z2,
                                                                q_xy2,q_xz2,q_yz2,
                                                                q_x, q_y, q_z,
                                                                q_xy, q_xz, q_yz,
                                                                sigma, alpha0,
                                                                width, height, depth);
        cudaCheckError( cudaDeviceSynchronize() );

        tgv_launch_divergence2<Pixel>(
                    q_x, q_y, q_z,
                    q_xy, q_xz, q_yz,
                    q_x2, q_y2, q_z2,
                    q_temp,
                    width, height, depth,
                    block_dimension,
                    grid_dimension,
                    grid_dimension_x,
                    grid_dimension_y,
                    grid_dimension_z);
        cudaCheckError( cudaDeviceSynchronize() );


        /*
            % primal update v
            v = v - tau * (nabla_second_t * q - p);
            v_bar = v + theta*(v - v_old);
            */

        tgv_kernel_part6<<<grid_dimension, block_dimension>>>(
                                                                v_x, v_y, v_z,
                                                                q_x2, q_y2, q_z2,
                                                                p_xx, p_yy, p_zz,
                                                                v_previous_x, v_previous_y, v_previous_z,
                                                                v_bar_x, v_bar_y, v_bar_z,
                                                                tau, theta,
                                                                width, height, depth);
        cudaCheckError( cudaDeviceSynchronize() );


        if(paint_iteration_interval > 0 &&
                iteration_index % paint_iteration_interval == 0) {
            printf("TVL2, iteration=%d / %d \n", iteration_index, iteration_count);
            bool stop = iteration_finished_callback(iteration_index, iteration_count, u);

            if(stop)
                break;
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
    tgv_launch_part32<Pixel>( depth,
                              v_bar_x, v_bar_y, v_bar_z,
                              v_previous_x, v_previous_y, v_previous_z,
                              v_x, v_y, v_z,
                              q_x, q_y, q_z,
                              q_xy, q_xz, q_yz,
                              q_x2, q_y2, q_z2,
                              q_xy2, q_xz2, q_yz2, q_temp);
    cudaFree(lambdas);

    return destination;
}

// generate the algorithm explicitly for...

template float* tgv2_l1_lambdas_launch(float* f_host,
float* lambdas_host,
uint width, uint height, uint depth,
float lambda_offset,
float lambda_factor,
uint iteration_count,
uint paint_iteration_interval,
IterationCallback<float> iteration_finished_callback,
float alpha0,
float alpha1);

template double* tgv2_l1_lambdas_launch(double* f_host,
double* lambdas_host,
uint width, uint height, uint depth,
double lambda_offset,
double lambda_factor,
uint iteration_count,
uint paint_iteration_interval,
IterationCallback<double> iteration_finished_callback,
double alpha0,
double alpha1);
