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


#include "tgv_common_2d.cu"
#include "tgv2_common_2d.cu"

#include "tgv2_common.cu"

template<typename Pixel>
__global__ void tgv_kernel_part4_tgv2_l1_2d(
        Pixel* p_x, Pixel* p_y,
        const Pixel tau, Pixel* u, Pixel* f,
        const Pixel lambda,
        Pixel* u_previous, const Pixel theta, Pixel* u_bar,
        const uint width, const uint height)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height)
        return;

    u[index] -= tau * (p_x[index] + p_y[index]);

    const Pixel tau_x_lambda = tau*lambda;
    const Pixel u_minus_f = u[index] - f[index];
    if(u_minus_f > tau_x_lambda)
        u[index] -= tau_x_lambda;
    else if(u_minus_f < -tau_x_lambda)
        u[index] += tau_x_lambda;
    else
        u[index] = f[index];

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
Pixel* tgv2_l1_launch_2d(Pixel* f_host,
                  uint width, uint height,
                  Pixel lambda,
                  uint iteration_count,
                  uint paint_iteration_interval,
                  IterationCallback<Pixel> iteration_finished_callback,
                  Pixel alpha0,
                  Pixel alpha1)
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

    tgv_launch_part1_2d<Pixel>(
                        width, height,
                        voxel_count,
                        block_dimension,
                        grid_dimension,
                        grid_dimension_x,
                        grid_dimension_y);

    Pixel* f, *u;
    Pixel* u_previous, *u_bar, *p_x, *p_y, *p_xx, *p_yy;

    tgv_launch_part2_2d<Pixel>(f_host,
                voxel_count,
                &f, &u,
                &u_previous, &u_bar, &p_x, &p_y, &p_xx, &p_yy);

    Pixel *v_x, *v_y;
    Pixel *v_bar_x, *v_bar_y;
    Pixel *v_previous_x, *v_previous_y;
    Pixel *q_x, *q_y, *q_xy;
    Pixel *q_x2, *q_y2, *q_xy2;
    Pixel * q_temp;

    tgv_launch_part22_2d<Pixel>(
                voxel_count,
                &v_previous_x, &v_previous_y,
                &v_bar_x, &v_bar_y,
                &v_x, &v_y,
                &q_x, &q_y,
                &q_xy,
                &q_x2, &q_y2,
                &q_xy2, &q_temp);

    // algorithm variables..
    const Pixel tau = 1.0 / std::sqrt(20.0);
    const Pixel sigma = tau;
    const Pixel theta = 1;

    // algorithm begin
    zeroInit_2d<<<grid_dimension, block_dimension>>>(
                                                    p_x, p_y,
                                                    p_xx, p_yy,
                                                    voxel_count);
    zeroInit_2d<<<grid_dimension, block_dimension>>>(
                                                    v_x, v_y,
                                                    v_bar_x, v_bar_y,
                                                    voxel_count);
    zeroInit2_2d<<<grid_dimension, block_dimension>>>(
                                                    q_x, q_y,
                                                    q_xy,
                                                    voxel_count);
    clone2<<<grid_dimension, block_dimension>>>(
                                                  f, u, u_bar, voxel_count);
    cudaCheckError( cudaDeviceSynchronize() );

    for(uint iteration_index = 0; iteration_index < iteration_count; iteration_index++)
    {
        tgv_launch_forward_differences_2d<Pixel>(u_bar,
                p_x,p_y,
                width, height,
                block_dimension,
                grid_dimension_x,
                grid_dimension_y);

        tgv_kernel_part22_2d<<<grid_dimension, block_dimension>>>( v_bar_x, v_bar_y,
                                                                p_x, p_y,
                                                                p_xx, p_yy,
                                                                sigma, alpha1, u_previous, u,
                                                                width, height);
        cudaCheckError( cudaDeviceSynchronize() );

        tgv_launch_backward_differences_2d<Pixel>(
                p_x, p_y,
                p_xx, p_yy,
                width, height,
                block_dimension,
                grid_dimension_x,
                grid_dimension_y);
        cudaCheckError( cudaDeviceSynchronize() );

        tgv_kernel_part4_tgv2_l1_2d<<<grid_dimension, block_dimension>>>(
                                                                p_x, p_y,
                                                                tau, u, f,
                                                                lambda,
                                                                u_previous, theta, u_bar,
                                                                width, height);
        cudaCheckError( cudaDeviceSynchronize() );

        /*
        % dual update q
        q = q + sigma*nabla_second*v_bar;
        norm_q = sqrt(q(1:N).^2 + q(N+1:2*N).^2 + q(2*N+1:3*N).^2 + ... % main diagonal
            2*q(3*N+1:4*N).^2 + 2*q(4*N+1:5*N).^2 + 2*q(5*N+1:6*N).^2); % off diagonal
        q = q./max(1, repmat(norm_q, 6, 1)/alpha0);

            v_old = v;
        */
        tgv_launch_gradient2_2d<Pixel>(
                v_bar_x, v_bar_y,
                q_x2,q_y2,
                q_xy2, q_temp,
                width, height,
                block_dimension,
                grid_dimension,
                grid_dimension_x,
                grid_dimension_y);

        tgv_kernel_part5_2d<<<grid_dimension, block_dimension>>>(
                v_x, v_y,
                v_previous_x, v_previous_y,
                q_x2,q_y2,
                q_xy2,
                q_x, q_y,
                q_xy,
                sigma, alpha0,
                width, height);
        cudaCheckError( cudaDeviceSynchronize() );

        tgv_launch_divergence2_2d<Pixel>(
                q_x, q_y,
                q_xy,
                q_x2, q_y2,
                q_temp,
                width, height,
                block_dimension,
                grid_dimension,
                grid_dimension_x,
                grid_dimension_y);
        cudaCheckError( cudaDeviceSynchronize() );


        /*
            % primal update v
            v = v - tau * (nabla_second_t * q - p);
            v_bar = v + theta*(v - v_old);
            */

        tgv_kernel_part6_2d<<<grid_dimension, block_dimension>>>(
                v_x, v_y,
                q_x2, q_y2,
                p_xx, p_yy,
                v_previous_x, v_previous_y,
                v_bar_x, v_bar_y,
                tau, theta,
                width, height);
        cudaCheckError( cudaDeviceSynchronize() );

        bool stop = tgv2_iteration_callback(
                    iteration_index, iteration_count, paint_iteration_interval,
                    u,
                    iteration_finished_callback, voxel_count);
        if(stop)
            break;
    }

    Pixel* destination = new Pixel[voxel_count];
    tgv_launch_part3_2d<Pixel>(
                destination,
                voxel_count,
                u_previous, u_bar,
                p_x, p_y,
                p_xx, p_yy,
                f, u);
    tgv_launch_part32_2d<Pixel>(
              v_bar_x, v_bar_y,
              v_previous_x, v_previous_y,
                v_x, v_y,
                q_x, q_y,
                q_xy,
              q_x2, q_y2,
              q_xy2, q_temp);

    return destination;
}

// generate the algorithm explicitly for...

template float* tgv2_l1_launch_2d(float* f_host,
uint width, uint height,
float lambda,
uint iteration_count,
uint paint_iteration_interval,
IterationCallback<float> iteration_finished_callback,
float alpha0,
float alpha1);

template double* tgv2_l1_launch_2d(double* f_host,
uint width, uint height,
double lambda,
uint iteration_count,
uint paint_iteration_interval,
IterationCallback<double> iteration_finished_callback,
double alpha0,
double alpha1);
