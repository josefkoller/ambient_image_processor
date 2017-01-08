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


#include "tgvk_masked_common_2d.cu"
#include "tgvk_common_2d.cu"

// k >= 3

template<typename Pixel>
Pixel* tgvk_l1_deshade_masked_2d_launch(Pixel* f_host,
  const uint width, const uint height,
  const Pixel lambda,
  const uint iteration_count,
  const uint paint_iteration_interval,
  const int cuda_block_dimension,
  DeshadeIterationCallback2D<Pixel> iteration_finished_callback,
  const uint order,
  const Pixel* alpha,
  Pixel** v_x_host, Pixel**v_y_host,

  IndexVector masked_pixel_indices,
  IndexVector left_edge_pixel_indices, IndexVector not_left_edge_pixel_indices,
  IndexVector right_edge_pixel_indices, IndexVector not_right_edge_pixel_indices,

  IndexVector top_edge_pixel_indices, IndexVector not_top_edge_pixel_indices,
  IndexVector bottom_edge_pixel_indices, IndexVector not_bottom_edge_pixel_indices)
{
    uint voxel_count;
    dim3 block_dimension;
    dim3 all_grid_dimension;
    dim3 left_grid_dimension, not_left_grid_dimension;
    dim3 right_grid_dimension, not_right_grid_dimension;
    dim3 top_grid_dimension, not_top_grid_dimension;
    dim3 bottom_grid_dimension, not_bottom_grid_dimension;
    dim3 masked_grid_dimension;

    tgv_launch_part1_masked_2d<Pixel>(
        width, height,
        voxel_count,
        block_dimension,

        left_edge_pixel_indices.size(), not_left_edge_pixel_indices.size(),
        right_edge_pixel_indices.size(), not_right_edge_pixel_indices.size(),
        top_edge_pixel_indices.size(), not_top_edge_pixel_indices.size(),
        bottom_edge_pixel_indices.size(), not_bottom_edge_pixel_indices.size(),
        masked_pixel_indices.size(),

        left_grid_dimension, not_left_grid_dimension,
        right_grid_dimension, not_right_grid_dimension,
        top_grid_dimension, not_top_grid_dimension,
        bottom_grid_dimension, not_bottom_grid_dimension,
        masked_grid_dimension, all_grid_dimension,

        cuda_block_dimension);

    NotConstIndex* left_indices; NotConstIndexCount left_indices_count;
    NotConstIndex* not_left_indices; NotConstIndexCount not_left_indices_count;
    NotConstIndex* right_indices; NotConstIndexCount right_indices_count;
    NotConstIndex* not_right_indices; NotConstIndexCount not_right_indices_count;
    NotConstIndex* top_indices; NotConstIndexCount top_indices_count;
    NotConstIndex* not_top_indices; NotConstIndexCount not_top_indices_count;
    NotConstIndex* bottom_indices; NotConstIndexCount bottom_indices_count;
    NotConstIndex* not_bottom_indices; NotConstIndexCount not_bottom_indices_count;
    NotConstIndex* masked_indices; NotConstIndexCount masked_indices_count;

    copyIndicesToDevice(left_edge_pixel_indices, &left_indices, left_indices_count);
    copyIndicesToDevice(not_left_edge_pixel_indices, &not_left_indices, not_left_indices_count);
    copyIndicesToDevice(right_edge_pixel_indices, &right_indices, right_indices_count);
    copyIndicesToDevice(not_right_edge_pixel_indices, &not_right_indices, not_right_indices_count);
    copyIndicesToDevice(top_edge_pixel_indices, &top_indices, top_indices_count);
    copyIndicesToDevice(not_top_edge_pixel_indices, &not_top_indices, not_top_indices_count);
    copyIndicesToDevice(bottom_edge_pixel_indices, &bottom_indices, bottom_indices_count);
    copyIndicesToDevice(not_bottom_edge_pixel_indices, &not_bottom_indices, not_bottom_indices_count);
    copyIndicesToDevice(masked_pixel_indices, &masked_indices, masked_indices_count);

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

    // order 3 -> 1x w, w_bar, w_previous, r, r2
    // order 4 -> 2x w, w_bar, w_previous, r, r2
    // order 5 -> 3x w, w_bar, w_previous, r, r2

    int k_minus_2 = order - 2;
    Pixel *w_x[k_minus_2], *w_y[k_minus_2], *w_xy[k_minus_2];
    Pixel *w_bar_x[k_minus_2], *w_bar_y[k_minus_2], *w_bar_xy[k_minus_2];
    Pixel *w_previous_x[k_minus_2], *w_previous_y[k_minus_2], *w_previous_xy[k_minus_2];
    Pixel *r_x[k_minus_2], *r_y[k_minus_2], *r_xy[k_minus_2];
    Pixel *r2_x[k_minus_2], *r2_y[k_minus_2], *r2_xy[k_minus_2];

    for(int i = 0; i < order - 2; i++)
    {
        tgv3_launch_part23_2d<Pixel>(
                    voxel_count,
                    &w_x[i], &w_y[i],
                    &w_xy[i],
                    &w_bar_x[i], &w_bar_y[i],
                    &w_bar_xy[i],
                    &w_previous_x[i], &w_previous_y[i],
                    &w_previous_xy[i],
                    &r_x[i], &r_y[i],
                    &r_xy[i],
                    &r2_x[i], &r2_y[i],
                    &r2_xy[i]);
    }

    // algorithm variables..
    const Pixel tau = 1.0 / std::sqrt(20.0);
    const Pixel sigma = tau;
    const Pixel theta = 1;

    // algorithm begin
    zeroInit_2d<<<all_grid_dimension, block_dimension>>>(
                                                    p_x, p_y,
                                                    p_xx, p_yy,
                                                    voxel_count);
    zeroInit_2d<<<all_grid_dimension, block_dimension>>>(
                                                    v_x, v_y,
                                                    v_bar_x, v_bar_y,
                                                    voxel_count);
    zeroInit_2d<<<all_grid_dimension, block_dimension>>>(
                                                    q_x, q_y,
                                                    q_xy, q_y,
                                                    voxel_count);

    for(int i = 0; i < order - 2; i++)
    {
        zeroInit2_2d<<<all_grid_dimension, block_dimension>>>(
                                                        w_x[i], w_y[i],
                                                        w_xy[i],
                                                        voxel_count);
        zeroInit2_2d<<<all_grid_dimension, block_dimension>>>(
                                                        w_bar_x[i], w_bar_y[i],
                                                        w_bar_xy[i],
                                                        voxel_count);
        zeroInit2_2d<<<all_grid_dimension, block_dimension>>>(
                                                        r_x[i], r_y[i],
                                                        r_xy[i],
                                                        voxel_count);
    }

    clone2<<<all_grid_dimension, block_dimension>>>(
                                                  f, u, u_bar, voxel_count);
    cudaCheckError( cudaDeviceSynchronize() );

    const Pixel tau_x_lambda = tau * lambda;
    for(uint iteration_index = 0; iteration_index < iteration_count; iteration_index++)
    {
      tgv_launch_forward_differences_masked_2d<Pixel>(u_bar,
        p_x, p_y,
        width,
        block_dimension,
        right_grid_dimension,
        not_right_grid_dimension,
        bottom_grid_dimension,
        not_bottom_grid_dimension,
        right_indices, right_indices_count,
        not_right_indices, not_right_indices_count,
        bottom_indices, bottom_indices_count,
        not_bottom_indices, not_bottom_indices_count);

        tgv_kernel_part22_masked_2d<<<masked_grid_dimension, block_dimension>>>(
            v_bar_x, v_bar_y,
            p_x, p_y,
            p_xx, p_yy,
            sigma, alpha[0], u_previous, u,
            masked_indices, masked_indices_count);
        cudaCheckError( cudaDeviceSynchronize() );


        tgv_launch_backward_differences_masked_2d<Pixel>(
          p_x, p_y,
          p_xx, p_yy,
          width,
          block_dimension,
          left_grid_dimension, not_left_grid_dimension,
          top_grid_dimension, not_top_grid_dimension,
          left_indices, left_indices_count,
          not_left_indices, not_left_indices_count,
          top_indices, top_indices_count,
          not_top_indices, not_top_indices_count);

        tgv_kernel_part4_tgv2_l1_masked_2d<<<masked_grid_dimension, block_dimension>>>(
            p_x, p_y,
            u, f,
            tau_x_lambda, tau,
            u_previous, theta, u_bar,
            masked_indices, masked_indices_count);
        cudaCheckError( cudaDeviceSynchronize() );


        // dual update q

        tgv_launch_gradient2_masked_2d(
          v_bar_x, v_bar_y,
          q_x2, q_y2,
          q_xy2, q_temp,
          width,
          block_dimension,
          masked_grid_dimension,
          left_grid_dimension,
          not_left_grid_dimension,
          top_grid_dimension,
          not_top_grid_dimension,
          masked_indices, masked_indices_count,
          left_indices, left_indices_count,
          not_left_indices, not_left_indices_count,
          top_indices, top_indices_count,
          not_top_indices, not_top_indices_count);

        tgv3_kernel_part5_masked_2d<<<masked_grid_dimension, block_dimension>>>(
                v_x, v_y,

                v_previous_x, v_previous_y,

                q_x2,q_y2,
                q_xy2,

                q_x, q_y,
                q_xy,

                w_bar_x[0], w_bar_y[0],
                w_bar_xy[0],

                sigma, alpha[1],
                masked_indices, masked_indices_count);
        cudaCheckError( cudaDeviceSynchronize() );

        tgv_launch_divergence2_masked_2d<Pixel>(
          q_x, q_y,
          q_xy,
          q_x2, q_y2,
          q_temp,
          width,
          block_dimension,
          right_grid_dimension,
          not_right_grid_dimension,
          bottom_grid_dimension,
          not_bottom_grid_dimension,
          masked_grid_dimension,
          right_indices, right_indices_count,
          not_right_indices, not_right_indices_count,
          bottom_indices, bottom_indices_count,
          not_bottom_indices, not_bottom_indices_count,
          masked_indices, masked_indices_count);

        // primal update v

        tgv_kernel_part6_masked_2d<<<masked_grid_dimension, block_dimension>>>(
                v_x, v_y,
                q_x2, q_y2,
                p_xx, p_yy,
                v_previous_x, v_previous_y,
                v_bar_x, v_bar_y,
                tau, theta,
                masked_indices, masked_indices_count);
        cudaCheckError( cudaDeviceSynchronize() );

        // dual update r

        for(int i = 0; i < order - 2; i++)
        {
            if(i % 2 == 1)
            {
                tgv_launch_gradient3_masked_2d<Pixel>(
                        w_bar_x[i], w_bar_y[i],
                        w_bar_xy[i],

                        r2_x[i], r2_y[i],
                        r2_xy[i],

                        q_temp,

                        width,

                        block_dimension,
                        right_grid_dimension, not_right_grid_dimension,
                        bottom_grid_dimension, not_bottom_grid_dimension,
                        masked_grid_dimension,

                        right_indices, right_indices_count,
                        not_right_indices, not_right_indices_count,
                        bottom_indices, bottom_indices_count,
                        not_bottom_indices, not_bottom_indices_count,
                        masked_indices, masked_indices_count);
            }
            else
            {
                tgv_launch_gradient3_backward_masked_2d<Pixel>(
                        w_bar_x[i], w_bar_y[i],
                        w_bar_xy[i],

                        r2_x[i], r2_y[i],
                        r2_xy[i],

                        q_temp,

                        width,
                        block_dimension,
                        masked_grid_dimension,
                        left_grid_dimension,
                        not_left_grid_dimension,
                        top_grid_dimension,
                        not_top_grid_dimension,
                        masked_indices, masked_indices_count,
                        left_indices, left_indices_count,
                        not_left_indices, not_left_indices_count,
                        top_indices, top_indices_count,
                        not_top_indices, not_top_indices_count);
            }

            if(i == order - 3)
            {
                tgv3_kernel_part52_masked_2d<<<masked_grid_dimension, block_dimension>>>(
                        r2_x[i], r2_y[i],
                        r2_xy[i],

                        r_x[i], r_y[i],
                        r_xy[i],

                        w_x[i], w_y[i],
                        w_xy[i],

                        w_previous_x[i], w_previous_y[i],
                        w_previous_xy[i],

                        sigma, alpha[i + 2],
                        masked_indices, masked_indices_count);
            }
            else
            {
                tgvk_kernel_part5_masked_2d<<<masked_grid_dimension, block_dimension>>>(
                         r2_x[i], r2_y[i],
                         r2_xy[i],

                         r_x[i], r_y[i],
                         r_xy[i],

                         w_x[i], w_y[i],
                         w_xy[i],

                         w_previous_x[i], w_previous_y[i],
                         w_previous_xy[i],

                         w_bar_x[i + 1], w_bar_y[i + 1],     // minus primal of next order
                         w_bar_xy[i + 1],

                         sigma, alpha[i + 2],
                        masked_indices, masked_indices_count);
            }
            cudaCheckError( cudaDeviceSynchronize() );

            if(i % 2 == 1)
            {
                tgv_launch_divergence3_masked_2d<Pixel>(
                        r_x[i], r_y[i],
                        r_xy[i],

                        r2_x[i], r2_y[i],
                        r2_xy[i],

                        q_temp,

                        width,

                        block_dimension,
                        masked_grid_dimension,

                        left_grid_dimension, not_left_grid_dimension,
                        top_grid_dimension, not_top_grid_dimension,

                        masked_indices, masked_indices_count,
                        left_indices, left_indices_count,
                        not_left_indices, not_left_indices_count,
                        top_indices, top_indices_count,
                        not_top_indices, not_top_indices_count);
            }
            else
            {
                tgv_launch_divergence3_forward_masked_2d<Pixel>(
                        r_x[i], r_y[i],
                        r_xy[i],

                        r2_x[i], r2_y[i],
                        r2_xy[i],

                        q_temp,

                        width,

                        block_dimension,
                        right_grid_dimension, not_right_grid_dimension,
                        bottom_grid_dimension, not_bottom_grid_dimension,
                        masked_grid_dimension,

                        right_indices, right_indices_count,
                        not_right_indices, not_right_indices_count,
                        bottom_indices, bottom_indices_count,
                        not_bottom_indices, not_bottom_indices_count,
                        masked_indices, masked_indices_count);
            }

            // primal update w

            Pixel* q_x_tilt = i == 0 ? q_x : r_x[i - 1]; // minus dual of previous order
            Pixel* q_y_tilt = i == 0 ? q_y : r_y[i - 1];
            Pixel* q_xy_tilt = i == 0 ? q_xy : r_xy[i - 1];

            tgv3_kernel_part62_masked_2d<<<masked_grid_dimension, block_dimension>>>(
                    r2_x[i], r2_y[i],
                    r2_xy[i],

                    q_x_tilt, q_y_tilt,
                    q_xy_tilt,

                    w_x[i], w_y[i],
                    w_xy[i],

                    w_previous_x[i], w_previous_y[i],
                    w_previous_xy[i],

                    w_bar_x[i], w_bar_y[i],
                    w_bar_xy[i],

                    tau, theta,
                    masked_indices, masked_indices_count);
            cudaCheckError( cudaDeviceSynchronize() );
        }


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

    for(int i = 0; i < order - 2; i++)
    {
        tgv3_launch_part33_2d<Pixel>(
                    w_x[i], w_y[i],
                    w_xy[i],
                    w_bar_x[i], w_bar_y[i],
                    w_bar_xy[i],
                    w_previous_x[i], w_previous_y[i],
                    w_previous_xy[i],
                    r_x[i], r_y[i],
                    r_xy[i],
                    r2_x[i], r2_y[i],
                    r2_xy[i]);
    }

    freeIndices_2d(left_indices,
                 not_left_indices,
                 right_indices,
                 not_right_indices,
                 top_indices,
                 not_top_indices,
                 bottom_indices,
                 not_bottom_indices,
                 masked_indices);

    return destination;
}

// generate the algorithm explicitly for...

template float* tgvk_l1_deshade_masked_2d_launch(float* f_host,
                      const uint width, const uint height,
                      const float lambda,
                      const uint iteration_count,
                      const uint paint_iteration_interval,
                      const int cuda_block_dimension,
                      DeshadeIterationCallback2D<float> iteration_finished_callback,
                      const uint order,
                      const float* alpha,
                      float** v_x_host, float**v_y_host,
                      IndexVector masked_pixel_indices,
                      IndexVector left_edge_pixel_indices, IndexVector not_left_edge_pixel_indices,
                      IndexVector right_edge_pixel_indices, IndexVector not_right_edge_pixel_indices,

                      IndexVector top_edge_pixel_indices, IndexVector not_top_edge_pixel_indices,
                      IndexVector bottom_edge_pixel_indices, IndexVector not_bottom_edge_pixel_indices);

template double* tgvk_l1_deshade_masked_2d_launch(double* f_host,
                      const uint width, const uint height,
                      const double lambda,
                      const uint iteration_count,
                      const uint paint_iteration_interval,
                      const int cuda_block_dimension,
                      DeshadeIterationCallback2D<double> iteration_finished_callback,
                      const uint order,
                      const double* alpha,
                      double** v_x_host, double**v_y_host,
                      IndexVector masked_pixel_indices,
                      IndexVector left_edge_pixel_indices, IndexVector not_left_edge_pixel_indices,
                      IndexVector right_edge_pixel_indices, IndexVector not_right_edge_pixel_indices,

                      IndexVector top_edge_pixel_indices, IndexVector not_top_edge_pixel_indices,
                      IndexVector bottom_edge_pixel_indices, IndexVector not_bottom_edge_pixel_indices);
