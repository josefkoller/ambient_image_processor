

#include "tgv2_l1.cu"

#include <functional>

template<typename Pixel>
using DeshadeIterationCallback = std::function<bool(uint iteration_index, uint iteration_count,
    Pixel* u, Pixel* v_x, Pixel* v_y, Pixel* v_z)>;

template<typename Pixel>
Pixel* tgv2_l1_deshade_launch(Pixel* f_host,
                  uint width, uint height, uint depth,
                  Pixel lambda,
                  uint iteration_count,
                  uint paint_iteration_interval,
                  DeshadeIterationCallback<Pixel> iteration_finished_callback,
                  Pixel alpha0,
                  Pixel alpha1, Pixel** v_x_host, Pixel**v_y_host, Pixel**v_z_host)
{
    /* MATLAB Code
     *
        nabla_second = [nabla_x, zeros_sparse, zeros_sparse, nabla_y/2, nabla_z/2, zeros_sparse;...
            zeros_sparse, nabla_y, zeros_sparse, nabla_x/2, zeros_sparse, nabla_z/2;...
            zeros_sparse, zeros_sparse, nabla_z, zeros_sparse, nabla_x/2, nabla_y/2]';

        nabla_second_t = [nabla_x, zeros_sparse, zeros_sparse, nabla_y, nabla_z, zeros_sparse;...
            zeros_sparse, nabla_y, zeros_sparse, nabla_x, zeros_sparse, nabla_z;...
            zeros_sparse, zeros_sparse, nabla_z, zeros_sparse, nabla_x, nabla_y];

        % fixed parameters
        L = sqrt(20);  % Lipschitz constant ? this is an approx. to normest()
        tau    = 1/L;
        sigma  = 1/L;

        theta  = 1;

        % initializations
        p = zeros(3*N, 1);
        v = zeros(3*N, 1);
        q = zeros(6*N, 1);

        u = f;
        u_bar = u;  % overrelaxation u
        v_bar = v;
        nabla_t = nabla';
        %nabla_second_t = nabla_second';

        for currIter = 1:maxIter
            u_old = u;
            v_old = v;

            % dual update p
            p = p + sigma*(nabla*u_bar - v_bar);
            norm_p  = sqrt(p(1:N).^2 + p(N+1:2*N).^2 +  p(2*N+1:3*N).^2);
            p = p./max(1,[norm_p; norm_p; norm_p]/alpha1);

            % dual update q
            q = q + sigma*nabla_second*v_bar;
            norm_q = sqrt(q(1:N).^2 + q(N+1:2*N).^2 + q(2*N+1:3*N).^2 + ... % main diagonal
                2*q(3*N+1:4*N).^2 + 2*q(4*N+1:5*N).^2 + 2*q(5*N+1:6*N).^2); % off diagonal
            q = q./max(1, repmat(norm_q, 6, 1)/alpha0);

            % primal update u
            u = u - tau * nabla_t * p;

            % projection of u
            index1 = (u - f) > tau*lambda;
            index2 = (u - f) < -tau*lambda;
            index3 = abs(u - f) <= tau*lambda;

            u(index1) = u(index1) - tau*lambda;
            u(index2) = u(index2) + tau*lambda;
            u(index3) = f(index3);

            % overrelaxation u
            u_bar = u + theta*(u - u_old);

            % primal update v
            v = v - tau * (nabla_second_t * q - p);
            v_bar = v + theta*(v - v_old);
            */
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

    Pixel *v_x, *v_y, *v_z;
    Pixel *v_bar_x, *v_bar_y, *v_bar_z;
    Pixel *v_previous_x, *v_previous_y, *v_previous_z;
    Pixel *q_x, *q_y, *q_z, *q_xy, *q_xz, *q_yz;
    Pixel *q_x2, *q_y2, *q_z2, *q_xy2, *q_xz2, *q_yz2;
    Pixel * q_temp;

    tgv_launch_part22<Pixel>(
                voxel_count, depth,
                &v_previous_x, &v_previous_y, &v_previous_z,
                &v_bar_x, &v_bar_y, &v_bar_z,
                &v_x, &v_y, &v_z,
                &q_x, &q_y, &q_z,
                &q_xy, &q_xz, &q_yz,
                &q_x2, &q_y2, &q_z2,
                &q_xy2, &q_xz2, &q_yz2, &q_temp);

    // algorithm variables..
    const Pixel tau = 1.0 / std::sqrt(20.0);
    const Pixel sigma = tau;
    const Pixel theta = 1;

    // algorithm begin
    zeroInit<<<grid_dimension, block_dimension>>>(
                                                    p_x, p_y, p_z,
                                                    p_xx, p_yy, p_zz,
                                                    voxel_count, depth);
    zeroInit<<<grid_dimension, block_dimension>>>(
                                                    v_x, v_y, v_z,
                                                    v_bar_x, v_bar_y, v_bar_z,
                                                    voxel_count, depth);
    zeroInit2<<<grid_dimension, block_dimension>>>(
                                                    q_x, q_y, q_z,
                                                    q_xy, q_xz, q_yz,
                                                    voxel_count, depth);
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

        tgv_kernel_part22<<<grid_dimension, block_dimension>>>( v_bar_x, v_bar_y, v_bar_z,
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

        tgv_kernel_part4_tgv2_l1<<<grid_dimension, block_dimension>>>(
                                                                p_x, p_y, p_z,
                                                                tau, u, f,
                                                                lambda,
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
            bool stop = iteration_finished_callback(iteration_index, iteration_count, u,
                                                    v_x, v_y, v_z);
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

    // copy v from the device to the host memory...
    *v_x_host = new Pixel[voxel_count];
    *v_y_host = new Pixel[voxel_count];
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMemcpy(*v_x_host, v_x, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaMemcpy(*v_y_host, v_y, size, cudaMemcpyDeviceToHost) );
    if(depth > 1)
    {
        *v_z_host = new Pixel[voxel_count];
        cudaCheckError( cudaMemcpy(*v_z_host, v_z, size, cudaMemcpyDeviceToHost) );
    }
    cudaCheckError( cudaDeviceSynchronize() );

    tgv_launch_part32<Pixel>( depth,
              v_bar_x, v_bar_y, v_bar_z,
              v_previous_x, v_previous_y, v_previous_z,
                v_x, v_y, v_z,
                q_x, q_y, q_z,
                q_xy, q_xz, q_yz,
              q_x2, q_y2, q_z2,
              q_xy2, q_xz2, q_yz2, q_temp);

    return destination;
}

// generate the algorithm explicitly for...

template float* tgv2_l1_deshade_launch(float* f_host,
uint width, uint height, uint depth,
float lambda,
uint iteration_count,
uint paint_iteration_interval,
DeshadeIterationCallback<float> iteration_finished_callback,
float alpha0,
float alpha1, float** v_x, float** v_y, float** v_z);

template double* tgv2_l1_deshade_launch(double* f_host,
uint width, uint height, uint depth,
double lambda,
uint iteration_count,
uint paint_iteration_interval,
DeshadeIterationCallback<double> iteration_finished_callback,
double alpha0,
double alpha1, double** v_x, double** v_y, double** v_z);
