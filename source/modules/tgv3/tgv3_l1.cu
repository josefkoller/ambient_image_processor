
#include "tgv2_l1.cu"
#include "tgv3_common.cu"

template<typename Pixel>
Pixel* tgv3_l1_launch(Pixel* f_host,
                  uint width, uint height, uint depth,
                  Pixel lambda,
                  uint iteration_count,
                  uint paint_iteration_interval,
                  IterationCallback<Pixel> iteration_finished_callback,
                  Pixel alpha0,
                  Pixel alpha1,
                  Pixel alpha2)
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

    Pixel *w_x, *w_y, *w_z, *w_xy, *w_xz, *w_yz;
    Pixel *w_bar_x, *w_bar_y, *w_bar_z, *w_bar_xy, *w_bar_xz, *w_bar_yz;
    Pixel *w_previous_x, *w_previous_y, *w_previous_z, *w_previous_xy, *w_previous_xz, *w_previous_yz;
    Pixel *r_x, *r_y, *r_z, *r_xy, *r_xz, *r_yz;
    Pixel *r2_x, *r2_y, *r2_z, *r2_xy, *r2_xz, *r2_yz;

    tgv3_launch_part23<Pixel>(
                voxel_count, depth,
                &w_x, &w_y, &w_z,
                &w_xy, &w_xz, &w_yz,
                &w_bar_x, &w_bar_y, &w_bar_z,
                &w_bar_xy, &w_bar_xz, &w_bar_yz,
                &w_previous_x, &w_previous_y, &w_previous_z,
                &w_previous_xy, &w_previous_xz, &w_previous_yz,
                &r_x, &r_y, &r_z,
                &r_xy, &r_xz, &r_yz,
                &r2_x, &r2_y, &r2_z,
                &r2_xy, &r2_xz, &r2_yz);

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

    zeroInit2<<<grid_dimension, block_dimension>>>(
                                                    w_x, w_y, w_z,
                                                    w_xy, w_xz, w_yz,
                                                    voxel_count);
    zeroInit2<<<grid_dimension, block_dimension>>>(
                                                    w_bar_x, w_bar_y, w_bar_z,
                                                    w_bar_xy, w_bar_xz, w_bar_yz,
                                                    voxel_count);
    zeroInit2<<<grid_dimension, block_dimension>>>(
                                                    r_x, r_y, r_z,
                                                    r_xy, r_xz, r_yz,
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

                w_bar_x, w_bar_y, w_bar_z,
                w_bar_xy, w_bar_xz, w_bar_yz,

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

        tgv_launch_gradient3<Pixel>(
                w_bar_x, w_bar_y, w_bar_z,
                w_bar_xy, w_bar_xz, w_bar_yz,

                r2_x, r2_y, r2_z,
                r2_xy, r2_xz, r2_yz,

                q_temp,

                width, height, depth,
                block_dimension,
                grid_dimension,
                grid_dimension_x,
                grid_dimension_y,
                grid_dimension_z);

        tgv3_kernel_part52<<<grid_dimension, block_dimension>>>(
                r2_x, r2_y, r2_z,
                r2_xy, r2_xz, r2_yz,

                r_x, r_y, r_z,
                r_xy, r_xz, r_yz,

                w_x, w_y, w_z,
                w_xy, w_xz, w_yz,

                w_previous_x, w_previous_y, w_previous_z,
                w_previous_xy, w_previous_xz, w_previous_yz,

                sigma, alpha2,
                width, height, depth);
        cudaCheckError( cudaDeviceSynchronize() );

        tgv_launch_divergence3<Pixel>(
                r_x, r_y, r_z,
                r_xy, r_xz, r_yz,

                r2_x, r2_y, r2_z,
                r2_xy, r2_xz, r2_yz,

                q_temp,

                width, height, depth,
                block_dimension,
                grid_dimension,
                grid_dimension_x,
                grid_dimension_y,
                grid_dimension_z);
        cudaCheckError( cudaDeviceSynchronize() );


        // primal update w

        tgv3_kernel_part62<<<grid_dimension, block_dimension>>>(
                r2_x, r2_y, r2_z,
                r2_xy, r2_xz, r2_yz,

                q_x, q_y, q_z,
                q_xy, q_xz, q_yz,

                w_x, w_y, w_z,
                w_xy, w_xz, w_yz,

                w_previous_x, w_previous_y, w_previous_z,
                w_previous_xy, w_previous_xz, w_previous_yz,

                w_bar_x, w_bar_y, w_bar_z,
                w_bar_xy, w_bar_xz, w_bar_yz,

                tau, theta,
                width, height, depth);
        cudaCheckError( cudaDeviceSynchronize() );


        if(paint_iteration_interval > 0 &&
                iteration_index % paint_iteration_interval == 0) {
            printf("TGV3L1, iteration=%d / %d \n", iteration_index, iteration_count);
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
    tgv3_launch_part33<Pixel>(
                depth,
                w_x, w_y, w_z,
                w_xy, w_xz, w_yz,
                w_bar_x, w_bar_y, w_bar_z,
                w_bar_xy, w_bar_xz, w_bar_yz,
                w_previous_x, w_previous_y, w_previous_z,
                w_previous_xy, w_previous_xz, w_previous_yz,
                r_x, r_y, r_z,
                r_xy, r_xz, r_yz,
                r2_x, r2_y, r2_z,
                r2_xy, r2_xz, r2_yz);

    return destination;
}

// generate the algorithm explicitly for...

template float* tgv3_l1_launch(float* f_host,
uint width, uint height, uint depth,
float lambda,
uint iteration_count,
uint paint_iteration_interval,
IterationCallback<float> iteration_finished_callback,
float alpha0,
float alpha1,
float alpha2);

template double* tgv3_l1_launch(double* f_host,
uint width, uint height, uint depth,
double lambda,
uint iteration_count,
uint paint_iteration_interval,
IterationCallback<double> iteration_finished_callback,
double alpha0,
double alpha1,
double alpha2);
