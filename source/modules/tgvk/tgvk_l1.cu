
#include "tgv3_l1.cu"
#include "tgvk_common.cu"

// k >= 3

template<typename Pixel>
Pixel* tgvk_l1_launch(Pixel* f_host,
                      const uint width, const uint height, const uint depth,
                      const Pixel lambda,
                      const uint iteration_count,
                      const uint paint_iteration_interval,
                      IterationCallback<Pixel> iteration_finished_callback,
                      const uint order,
                      const Pixel* alpha)
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

    // order 3 -> 1x w, w_bar, w_previous, r, r2
    // order 4 -> 2x w, w_bar, w_previous, r, r2
    // order 5 -> 3x w, w_bar, w_previous, r, r2

    int k_minus_2 = order - 2;
    Pixel *w_x[k_minus_2], *w_y[k_minus_2], *w_z[k_minus_2], *w_xy[k_minus_2], *w_xz[k_minus_2], *w_yz[k_minus_2];
    Pixel *w_bar_x[k_minus_2], *w_bar_y[k_minus_2], *w_bar_z[k_minus_2], *w_bar_xy[k_minus_2], *w_bar_xz[k_minus_2], *w_bar_yz[k_minus_2];
    Pixel *w_previous_x[k_minus_2], *w_previous_y[k_minus_2], *w_previous_z[k_minus_2], *w_previous_xy[k_minus_2], *w_previous_xz[k_minus_2], *w_previous_yz[k_minus_2];
    Pixel *r_x[k_minus_2], *r_y[k_minus_2], *r_z[k_minus_2], *r_xy[k_minus_2], *r_xz[k_minus_2], *r_yz[k_minus_2];
    Pixel *r2_x[k_minus_2], *r2_y[k_minus_2], *r2_z[k_minus_2], *r2_xy[k_minus_2], *r2_xz[k_minus_2], *r2_yz[k_minus_2];

    for(int i = 0; i < order - 2; i++)
    {
        tgv3_launch_part23<Pixel>(
                    voxel_count, depth,
                    &w_x[i], &w_y[i], &w_z[i],
                    &w_xy[i], &w_xz[i], &w_yz[i],
                    &w_bar_x[i], &w_bar_y[i], &w_bar_z[i],
                    &w_bar_xy[i], &w_bar_xz[i], &w_bar_yz[i],
                    &w_previous_x[i], &w_previous_y[i], &w_previous_z[i],
                    &w_previous_xy[i], &w_previous_xz[i], &w_previous_yz[i],
                    &r_x[i], &r_y[i], &r_z[i],
                    &r_xy[i], &r_xz[i], &r_yz[i],
                    &r2_x[i], &r2_y[i], &r2_z[i],
                    &r2_xy[i], &r2_xz[i], &r2_yz[i]);
    }

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

    for(int i = 0; i < order - 2; i++)
    {
        zeroInit2<<<grid_dimension, block_dimension>>>(
                                                        w_x[i], w_y[i], w_z[i],
                                                        w_xy[i], w_xz[i], w_yz[i],
                                                        voxel_count);
        zeroInit2<<<grid_dimension, block_dimension>>>(
                                                        w_bar_x[i], w_bar_y[i], w_bar_z[i],
                                                        w_bar_xy[i], w_bar_xz[i], w_bar_yz[i],
                                                        voxel_count);
        zeroInit2<<<grid_dimension, block_dimension>>>(
                                                        r_x[i], r_y[i], r_z[i],
                                                        r_xy[i], r_xz[i], r_yz[i],
                                                        voxel_count);
    }

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
                                                                sigma, alpha[0], u_previous, u,
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

        // dual update q

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

        tgv3_kernel_part5<<<grid_dimension, block_dimension>>>(
                v_x, v_y, v_z,
                v_previous_x, v_previous_y, v_previous_z,
                q_x2,q_y2,q_z2,
                q_xy2,q_xz2,q_yz2,
                q_x, q_y, q_z,
                q_xy, q_xz, q_yz,

                w_bar_x[0], w_bar_y[0], w_bar_z[0],
                w_bar_xy[0], w_bar_xz[0], w_bar_yz[0],

                sigma, alpha[1],
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


        // primal update v

        tgv_kernel_part6<<<grid_dimension, block_dimension>>>(
                v_x, v_y, v_z,
                q_x2, q_y2, q_z2,
                p_xx, p_yy, p_zz,
                v_previous_x, v_previous_y, v_previous_z,
                v_bar_x, v_bar_y, v_bar_z,
                tau, theta,
                width, height, depth);
        cudaCheckError( cudaDeviceSynchronize() );

        // dual update r

        for(int i = 0; i < order - 2; i++)
        {
            if(i % 2 == 1)
            {
                tgv_launch_gradient3<Pixel>(
                        w_bar_x[i], w_bar_y[i], w_bar_z[i],
                        w_bar_xy[i], w_bar_xz[i], w_bar_yz[i],

                        r2_x[i], r2_y[i], r2_z[i],
                        r2_xy[i], r2_xz[i], r2_yz[i],

                        q_temp,

                        width, height, depth,
                        block_dimension,
                        grid_dimension,
                        grid_dimension_x,
                        grid_dimension_y,
                        grid_dimension_z);
            }
            else
            {
                tgv_launch_gradient3_backward<Pixel>(
                        w_bar_x[i], w_bar_y[i], w_bar_z[i],
                        w_bar_xy[i], w_bar_xz[i], w_bar_yz[i],

                        r2_x[i], r2_y[i], r2_z[i],
                        r2_xy[i], r2_xz[i], r2_yz[i],

                        q_temp,

                        width, height, depth,
                        block_dimension,
                        grid_dimension,
                        grid_dimension_x,
                        grid_dimension_y,
                        grid_dimension_z);

            }

            if(i == order - 3)
            {
                tgv3_kernel_part52<<<grid_dimension, block_dimension>>>(
                        r2_x[i], r2_y[i], r2_z[i],
                        r2_xy[i], r2_xz[i], r2_yz[i],

                        r_x[i], r_y[i], r_z[i],
                        r_xy[i], r_xz[i], r_yz[i],

                        w_x[i], w_y[i], w_z[i],
                        w_xy[i], w_xz[i], w_yz[i],

                        w_previous_x[i], w_previous_y[i], w_previous_z[i],
                        w_previous_xy[i], w_previous_xz[i], w_previous_yz[i],

                        sigma, alpha[i + 2],
                        width, height, depth);
            }
            else
            {
                tgvk_kernel_part5<<<grid_dimension, block_dimension>>>(
                         r2_x[i], r2_y[i], r2_z[i],
                         r2_xy[i], r2_xz[i], r2_yz[i],

                         r_x[i], r_y[i], r_z[i],
                         r_xy[i], r_xz[i], r_yz[i],

                         w_x[i], w_y[i], w_z[i],
                         w_xy[i], w_xz[i], w_yz[i],

                         w_previous_x[i], w_previous_y[i], w_previous_z[i],
                         w_previous_xy[i], w_previous_xz[i], w_previous_yz[i],

                         w_bar_x[i + 1], w_bar_y[i + 1], w_bar_z[i + 1],      // minus primal of next order
                         w_bar_xy[i + 1], w_bar_xz[i + 1], w_bar_yz[i + 1],

                         sigma, alpha[i + 2],
                         width, height, depth);
            }
            cudaCheckError( cudaDeviceSynchronize() );

            if(i % 2 == 1)
            {
                tgv_launch_divergence3<Pixel>(
                        r_x[i], r_y[i], r_z[i],
                        r_xy[i], r_xz[i], r_yz[i],

                        r2_x[i], r2_y[i], r2_z[i],
                        r2_xy[i], r2_xz[i], r2_yz[i],

                        q_temp,

                        width, height, depth,
                        block_dimension,
                        grid_dimension,
                        grid_dimension_x,
                        grid_dimension_y,
                        grid_dimension_z);
            }
            else
            {
                tgv_launch_divergence3_forward<Pixel>(
                        r_x[i], r_y[i], r_z[i],
                        r_xy[i], r_xz[i], r_yz[i],

                        r2_x[i], r2_y[i], r2_z[i],
                        r2_xy[i], r2_xz[i], r2_yz[i],

                        q_temp,

                        width, height, depth,
                        block_dimension,
                        grid_dimension,
                        grid_dimension_x,
                        grid_dimension_y,
                        grid_dimension_z);
            }


            // primal update w

            Pixel* q_x_tilt = i == 0 ? q_x : r_x[i - 1]; // minus dual of previous order
            Pixel* q_y_tilt = i == 0 ? q_y : r_y[i - 1];
            Pixel* q_z_tilt = i == 0 ? q_z : r_z[i - 1];
            Pixel* q_xy_tilt = i == 0 ? q_xy : r_xy[i - 1];
            Pixel* q_xz_tilt = i == 0 ? q_xz : r_xz[i - 1];
            Pixel* q_yz_tilt = i == 0 ? q_yz : r_yz[i - 1];

            tgv3_kernel_part62<<<grid_dimension, block_dimension>>>(
                    r2_x[i], r2_y[i], r2_z[i],
                    r2_xy[i], r2_xz[i], r2_yz[i],

                    q_x_tilt, q_y_tilt, q_z_tilt,
                    q_xy_tilt, q_xz_tilt, q_yz_tilt,

                    w_x[i], w_y[i], w_z[i],
                    w_xy[i], w_xz[i], w_yz[i],

                    w_previous_x[i], w_previous_y[i], w_previous_z[i],
                    w_previous_xy[i], w_previous_xz[i], w_previous_yz[i],

                    w_bar_x[i], w_bar_y[i], w_bar_z[i],
                    w_bar_xy[i], w_bar_xz[i], w_bar_yz[i],

                    tau, theta,
                    width, height, depth);
            cudaCheckError( cudaDeviceSynchronize() );
        }


        if(paint_iteration_interval > 0 &&
                iteration_index % paint_iteration_interval == 0) {
            printf("tgvkL1, iteration=%d / %d \n", iteration_index, iteration_count);
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

    for(int i = 0; i < order - 2; i++)
    {
        tgv3_launch_part33<Pixel>(
                    depth,
                    w_x[i], w_y[i], w_z[i],
                    w_xy[i], w_xz[i], w_yz[i],
                    w_bar_x[i], w_bar_y[i], w_bar_z[i],
                    w_bar_xy[i], w_bar_xz[i], w_bar_yz[i],
                    w_previous_x[i], w_previous_y[i], w_previous_z[i],
                    w_previous_xy[i], w_previous_xz[i], w_previous_yz[i],
                    r_x[i], r_y[i], r_z[i],
                    r_xy[i], r_xz[i], r_yz[i],
                    r2_x[i], r2_y[i], r2_z[i],
                    r2_xy[i], r2_xz[i], r2_yz[i]);
    }

    return destination;
}

// generate the algorithm explicitly for...

template float* tgvk_l1_launch(float* f_host,
                      const uint width, const uint height, const uint depth,
                      const float lambda,
                      const uint iteration_count,
                      const uint paint_iteration_interval,
                      IterationCallback<float> iteration_finished_callback,
                      const uint order,
                      const float* alpha);

template double* tgvk_l1_launch(double* f_host,
                      const uint width, const uint height, const uint depth,
                      const double lambda,
                      const uint iteration_count,
                      const uint paint_iteration_interval,
                      IterationCallback<double> iteration_finished_callback,
                      const uint order,
                      const double* alpha);

