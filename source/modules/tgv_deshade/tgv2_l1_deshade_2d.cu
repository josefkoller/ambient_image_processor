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



#include "tgv2_l1_2d.cu"

#include <functional>

template<typename Pixel>
Pixel* tgv2_l1_deshade_launch_2d(Pixel* f_host,
                  uint width, uint height,
                  Pixel lambda,
                  uint iteration_count,
                  uint paint_iteration_interval,
                  DeshadeIterationCallback2D<Pixel> iteration_finished_callback,
                  Pixel alpha0,
                  Pixel alpha1, Pixel** v_x_host, Pixel**v_y_host)
{
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
    clone2<<<grid_dimension, block_dimension>>>(f, u, u_bar, voxel_count);
    cudaCheckError( cudaDeviceSynchronize() );

    for(uint iteration_index = 0; iteration_index < iteration_count; iteration_index++)
    {
        tgv_launch_forward_differences_2d<Pixel>(u_bar,
                p_x, p_y,
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


        bool stop = tgv2_deshade_iteration_callback_2d(
                    iteration_index, iteration_count, paint_iteration_interval,
                    u, v_x, v_y,
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

    // copy v from the device to the host memory...
    *v_x_host = new Pixel[voxel_count];
    *v_y_host = new Pixel[voxel_count];
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMemcpy(*v_x_host, v_x, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaMemcpy(*v_y_host, v_y, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaDeviceSynchronize() );

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

template float* tgv2_l1_deshade_launch_2d(float* f_host,
uint width, uint height,
float lambda,
uint iteration_count,
uint paint_iteration_interval,
DeshadeIterationCallback2D<float> iteration_finished_callback,
float alpha0,
float alpha1, float** v_x, float** v_y);

template double* tgv2_l1_deshade_launch_2d(double* f_host,
uint width, uint height,
double lambda,
uint iteration_count,
uint paint_iteration_interval,
DeshadeIterationCallback2D<double> iteration_finished_callback,
double alpha0,
double alpha1, double** v_x, double** v_y);
