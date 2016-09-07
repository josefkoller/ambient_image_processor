

#include "tgv2_masked_common_2d.cu"
#include "tgv2_common_2d.cu"

template<typename Pixel>
Pixel* tgv2_l1_deshade_masked_2d_launch(Pixel* f_host,
  uint width, uint height,
  Pixel lambda,
  uint iteration_count,
  uint paint_iteration_interval,
  const int cuda_block_dimension,
  DeshadeIterationCallback2D<Pixel> iteration_finished_callback,
  Pixel alpha0,
  Pixel alpha1, Pixel** v_x_host, Pixel**v_y_host,

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
    clone2<<<all_grid_dimension, block_dimension>>>(f, u, u_bar, voxel_count);

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
            sigma, alpha1, u_previous, u,
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

        tgv_kernel_part5_masked_2d<<<masked_grid_dimension, block_dimension>>>(
          v_x, v_y,
          v_previous_x, v_previous_y,
          q_x2,q_y2,
          q_xy2,
          q_x, q_y,
          q_xy,
          sigma, alpha0,
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

        tgv_kernel_part6_masked_2d<<<masked_grid_dimension, block_dimension>>>(
                v_x, v_y,
                q_x2, q_y2,
                p_xx, p_yy,
                v_previous_x, v_previous_y,
                v_bar_x, v_bar_y,
                tau, theta,
                masked_indices, masked_indices_count);
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


template
float* tgv2_l1_deshade_masked_2d_launch(float* f_host,
  uint width, uint height,
  float lambda,
  uint iteration_count,
  uint paint_iteration_interval,
  const int cuda_block_dimension,
  DeshadeIterationCallback2D<float> iteration_finished_callback,
  float alpha0,
  float alpha1, float** v_x_host, float**v_y_host,

  IndexVector masked_pixel_indices,
  IndexVector left_edge_pixel_indices, IndexVector not_left_edge_pixel_indices,
  IndexVector right_edge_pixel_indices, IndexVector not_right_edge_pixel_indices,

  IndexVector top_edge_pixel_indices, IndexVector not_top_edge_pixel_indices,
  IndexVector bottom_edge_pixel_indices, IndexVector not_bottom_edge_pixel_indices);

template
double* tgv2_l1_deshade_masked_2d_launch(double* f_host,
  uint width, uint height,
  double lambda,
  uint iteration_count,
  uint paint_iteration_interval,
  const int cuda_block_dimension,
  DeshadeIterationCallback2D<double> iteration_finished_callback,
  double alpha0,
  double alpha1, double** v_x_host, double**v_y_host,

  IndexVector masked_pixel_indices,
  IndexVector left_edge_pixel_indices, IndexVector not_left_edge_pixel_indices,
  IndexVector right_edge_pixel_indices, IndexVector not_right_edge_pixel_indices,

  IndexVector top_edge_pixel_indices, IndexVector not_top_edge_pixel_indices,
  IndexVector bottom_edge_pixel_indices, IndexVector not_bottom_edge_pixel_indices);
