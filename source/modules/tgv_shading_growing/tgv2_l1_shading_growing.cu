
#include "tgv2_l1.cu"
#include "shading_growing.cu"

template<typename Pixel>
Pixel* tgv2_l1_shading_growing(Pixel* f_host,
                               uint width, uint height, uint depth,
                               Pixel lambda,
                               uint iteration_count,
                               uint paint_iteration_interval,
                               IterationCallback<Pixel> iteration_finished_callback,
                               Pixel alpha0,
                               Pixel alpha1,
                               Pixel lower_threshold,
                               Pixel upper_threshold,
                               Pixel* non_local_gradient_kernel_host,
                               uint non_local_gradient_kernel_size
                               )
{
    // initialize
    PixelIndex dimensions(width, height, depth);
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

    // additional images ...
    Pixel* u_target, *label_image;
    size_t size = sizeof(Pixel) * voxel_count;
    cudaCheckError( cudaMallocManaged(&u_target, size) )
    cudaCheckError( cudaMallocManaged(&label_image, size) )

    Pixel* non_local_gradient_kernel;
    size_t kernel_size = sizeof(Pixel) * (non_local_gradient_kernel_size*non_local_gradient_kernel_size*non_local_gradient_kernel_size);
    cudaCheckError( cudaMallocManaged(&non_local_gradient_kernel, kernel_size) )
    cudaCheckError( cudaMemcpy(non_local_gradient_kernel, non_local_gradient_kernel_host, kernel_size, cudaMemcpyHostToDevice) )

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

    // start iterations

    for(uint iteration_index = 0; iteration_index < iteration_count; iteration_index++)
    {
        shading_growing(u,
                        dimensions,
                        lower_threshold,
                        upper_threshold,
                        u_target,
                        label_image,

                        non_local_gradient_kernel,
                        non_local_gradient_kernel_size,

                        p_x,
                        block_dimension,
                        grid_dimension,
                        grid_dimension_x,
                        grid_dimension_y,
                        grid_dimension_z);

        /* // return the segmentation result...
        for(int i = 0; i < voxel_count; i++)
            u[i] = label_image[i];
        break;
        */


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
                                                                tau, u, u_target,
                                                                lambda,
                                                                u_previous, theta, u_bar,
                                                                width, height, depth);
        cudaCheckError( cudaDeviceSynchronize() );

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

    // copy result, clean up

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
    cudaFree(u_target);
    cudaFree(label_image);
    cudaFree(non_local_gradient_kernel);

    return destination;
}

// generate the algorithm explicitly for...

template float* tgv2_l1_shading_growing(float* f_host,
uint width, uint height, uint depth,
float lambda,
uint iteration_count,
uint paint_iteration_interval,
IterationCallback<float> iteration_finished_callback,
float alpha0,
float alpha1,
float lower_threshold,
float upper_threshold,
float* non_local_gradient_kernel_host,
uint non_local_gradient_kernel_size);

template double* tgv2_l1_shading_growing(double* f_host,
uint width, uint height, uint depth,
double lambda,
uint iteration_count,
uint paint_iteration_interval,
IterationCallback<double> iteration_finished_callback,
double alpha0,
double alpha1,
double lower_threshold,
double upper_threshold,
double* non_local_gradient_kernel_host,
uint non_local_gradient_kernel_size);

