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
#include "tgv3_common_2d.cu"

#include <functional>

template<typename Pixel>
Pixel* tgv3_l1_deshade_launch_2d(Pixel* f_host,
                  uint width, uint height,
                  Pixel lambda,
                  uint iteration_count,
                  uint paint_iteration_interval,
                  DeshadeIterationCallback2D<Pixel> iteration_finished_callback,
                  Pixel alpha0,
                  Pixel alpha1,
                  Pixel alpha2,
                  Pixel** v_x_host, Pixel**v_y_host)
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

    Pixel *w_x, *w_y, *w_xy;
    Pixel *w_bar_x, *w_bar_y, *w_bar_xy;
    Pixel *w_previous_x, *w_previous_y, *w_previous_xy;
    Pixel *r_x, *r_y, *r_xy;
    Pixel *r2_x, *r2_y, *r2_xy;

    tgv3_launch_part23_2d<Pixel>(
                voxel_count,
                &w_x, &w_y,
                &w_xy,
                &w_bar_x, &w_bar_y,
                &w_bar_xy,
                &w_previous_x, &w_previous_y,
                &w_previous_xy,
                &r_x, &r_y,
                &r_xy,
                &r2_x, &r2_y,
                &r2_xy);

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

    zeroInit2_2d<<<grid_dimension, block_dimension>>>(
                                                    w_x, w_y,
                                                    w_xy,
                                                    voxel_count);
    zeroInit2_2d<<<grid_dimension, block_dimension>>>(
                                                    w_bar_x, w_bar_y,
                                                    w_bar_xy,
                                                    voxel_count);
    zeroInit2_2d<<<grid_dimension, block_dimension>>>(
                                                    r_x, r_y,
                                                    r_xy,
                                                    voxel_count);

    clone2<<<grid_dimension, block_dimension>>>(f, u, u_bar, voxel_count);
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

        // dual update q

        tgv_launch_gradient2_2d<Pixel>(
                v_bar_x, v_bar_y,
                q_x2,q_y2,
                q_xy2, q_temp,
                width, height,
                block_dimension,
                grid_dimension,
                grid_dimension_x,
                grid_dimension_y);

        tgv3_kernel_part5_2d<<<grid_dimension, block_dimension>>>(
                v_x, v_y,
                v_previous_x, v_previous_y,
                q_x2,q_y2,
                q_xy2,
                q_x, q_y,
                q_xy,

                w_bar_x, w_bar_y,
                w_bar_xy,

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


        // primal update v

        tgv_kernel_part6_2d<<<grid_dimension, block_dimension>>>(
                v_x, v_y,
                q_x2, q_y2,
                p_xx, p_yy,
                v_previous_x, v_previous_y,
                v_bar_x, v_bar_y,
                tau, theta,
                width, height);
        cudaCheckError( cudaDeviceSynchronize() );

        // dual update r

        tgv_launch_gradient3_2d<Pixel>(
                w_bar_x, w_bar_y,
                w_bar_xy,

                r2_x, r2_y,
                r2_xy,

                q_temp,

                width, height,
                block_dimension,
                grid_dimension,
                grid_dimension_x,
                grid_dimension_y);

        tgv3_kernel_part52_2d<<<grid_dimension, block_dimension>>>(
                r2_x, r2_y,
                r2_xy,

                r_x, r_y,
                r_xy,

                w_x, w_y,
                w_xy,

                w_previous_x, w_previous_y,
                w_previous_xy,

                sigma, alpha2,
                width, height);
        cudaCheckError( cudaDeviceSynchronize() );

        tgv_launch_divergence3_2d<Pixel>(
                r_x, r_y,
                r_xy,

                r2_x, r2_y,
                r2_xy,

                q_temp,

                width, height,
                block_dimension,
                grid_dimension,
                grid_dimension_x,
                grid_dimension_y);
        cudaCheckError( cudaDeviceSynchronize() );


        // primal update w

        tgv3_kernel_part62_2d<<<grid_dimension, block_dimension>>>(
                r2_x, r2_y,
                r2_xy,

                q_x, q_y,
                q_xy,

                w_x, w_y,
                w_xy,

                w_previous_x, w_previous_y,
                w_previous_xy,

                w_bar_x, w_bar_y,
                w_bar_xy,

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

    tgv3_launch_part33_2d<Pixel>(
                w_x, w_y,
                w_xy,
                w_bar_x, w_bar_y,
                w_bar_xy,
                w_previous_x, w_previous_y,
                w_previous_xy,
                r_x, r_y,
                r_xy,
                r2_x, r2_y,
                r2_xy);

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

template float* tgv3_l1_deshade_launch_2d(float* f_host,
uint width, uint height,
float lambda,
uint iteration_count,
uint paint_iteration_interval,
DeshadeIterationCallback2D<float> iteration_finished_callback,
float alpha0,
float alpha1,
float alpha2,
float** v_x, float** v_y);

template double* tgv3_l1_deshade_launch_2d(double* f_host,
uint width, uint height,
double lambda,
uint iteration_count,
uint paint_iteration_interval,
DeshadeIterationCallback2D<double> iteration_finished_callback,
double alpha0,
double alpha1,
double alpha2,
double** v_x, double** v_y);
