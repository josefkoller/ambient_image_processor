
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/adjacent_difference.h>
#include <algorithm>

#include <stdio.h>

#include "ThrustImage.cuh"
#include "thrust_operators.cuh"

template<typename PixelVector>
void print(ThrustImage<PixelVector>* ThrustImage, std::string title)
{
    std::cout << "ThrustImage: " << title << std::endl;
    for(int z = 0; z < ThrustImage->depth; z++)
    {
        for(int y = 0; y < ThrustImage->height; y++)
        {
            for(int x = 0; x < ThrustImage->width; x++)
            {
                std::cout << ThrustImage->getPixel(x,y, z) << "\t";
            }
            std::cout << std::endl;
        }
    }
}

template<typename PixelVector>
__host__ __device__
ThrustImage<PixelVector>* filter(ThrustImage<PixelVector>* f,
                                 const Pixel lambda,
                                 const unsigned int iteration_count,
                                 const unsigned int paint_iteration_interval,
                                 std::function<void(uint iteration_index, uint iteration_count, ThrustImage<PixelVector>*)> iteration_finished_callback
                                 )
{
    typedef ThrustImage<PixelVector> ThrustThrustImage;


#if THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_OMP
    omp_set_num_threads(NUMBER_OF_THREADS);
#endif

    printf("TVL2, lambda=%f \n", lambda);

    /*
  f = new ThrustThrustImage(3,3);
  f->setPixel(0, 0, 1);
  f->setPixel(1, 0, 3);
  f->setPixel(2, 0, 7);

  f->setPixel(0, 1, 15);
  f->setPixel(1, 1, 25);
  f->setPixel(2, 1, 42);

  f->setPixel(0, 2, 85);
  f->setPixel(1, 2, 166);
  f->setPixel(2, 2, 512);
*/


    const Pixel sqrt_8 = std::sqrt(8.0);
    Pixel tau = 1.0 / sqrt_8;
    Pixel sigma = tau;

    Pixel theta = 1.0; // will be used later

    ThrustThrustImage* u = f->clone();
    ThrustThrustImage* p_x_temp = u->clone_uninitialized(); // memory allocation
    ThrustThrustImage* p_y_temp = u->clone_uninitialized(); // memory allocation
    ThrustThrustImage* p_x = u->clone_initialized(0); // memory allocation
    ThrustThrustImage* p_y = u->clone_initialized(0); // memory allocation

    ThrustThrustImage* p_z_temp;
    ThrustThrustImage* p_z;
    if(u->depth > 1) {
        p_z_temp = u->clone_uninitialized(); // memory allocation
        p_z = u->clone_initialized(0); // memory allocation
    }


    ThrustThrustImage* divergence_p = u->clone_uninitialized(); // memory allocation
    ThrustThrustImage* u_bar = u->clone(); // memory allocation
    ThrustThrustImage* u_previous = u->clone_uninitialized(); // memory allocation

    ThrustThrustImage* p_magnitude = u->clone_uninitialized(); // memory allocation

    for(uint iteration_index = 0; iteration_index < iteration_count; iteration_index++)
    {
        /* matlab primal dual TVL2
       *
          u_old = u;

          % dual update
          p = p + sigma*nabla*u_bar;
          norm_p = sqrt(p(1:N).^2 + p(N+1:2*N).^2);
          p = p./max(1,[norm_p; norm_p]);

          % primal update
          u = u_old - tau * nabla_t * p;
          u = (u + tau * lambda .* f)/(1 + tau * lambda);

          % overrelaxation
          u_bar = u + theta*(u - u_old);
      *
      */


        // u_old = u;
        u_previous->set_pixel_data_of(u);

        //p = p + sigma*nabla*u_bar;
        u_bar->forward_difference_x(p_x_temp);
        u_bar->forward_difference_y(p_y_temp);

        if(u->depth > 1) {
            u_bar->forward_difference_z(p_z_temp);
        }

        thrust::transform(p_x_temp->pixel_rows.begin(), p_x_temp->pixel_rows.end(),
                          p_x->pixel_rows.begin(), p_x_temp->pixel_rows.begin(),
                          MultiplyByConstantAndAddOperation<Pixel>(sigma) );
        thrust::transform(p_y_temp->pixel_rows.begin(), p_y_temp->pixel_rows.end(),
                          p_y->pixel_rows.begin(), p_y_temp->pixel_rows.begin(),
                          MultiplyByConstantAndAddOperation<Pixel>(sigma) );

        if(u->depth > 1) {
            thrust::transform(p_z_temp->pixel_rows.begin(), p_z_temp->pixel_rows.end(),
                              p_z->pixel_rows.begin(), p_z_temp->pixel_rows.begin(),
                              MultiplyByConstantAndAddOperation<Pixel>(sigma) );
        }


        // norm_p = sqrt(p(1:N).^2 + p(N+1:2*N).^2);
        thrust::transform(p_x_temp->pixel_rows.begin(), p_x_temp->pixel_rows.end(),
                          p_x->pixel_rows.begin(),
                          SquareOperation<Pixel>() );
        thrust::transform(p_y_temp->pixel_rows.begin(), p_y_temp->pixel_rows.end(),
                          p_y->pixel_rows.begin(),
                          SquareOperation<Pixel>() );

        if(u->depth > 1) {
            thrust::transform(p_z_temp->pixel_rows.begin(), p_z_temp->pixel_rows.end(),
                              p_z->pixel_rows.begin(),
                              SquareOperation<Pixel>() );
        }

        thrust::transform(p_x->pixel_rows.begin(), p_x->pixel_rows.end(),
                          p_y->pixel_rows.begin(), p_magnitude->pixel_rows.begin(),
                          thrust::plus<Pixel>() );

        if(u->depth > 1) {
            thrust::transform(p_magnitude->pixel_rows.begin(), p_magnitude->pixel_rows.end(),
                              p_z->pixel_rows.begin(), p_magnitude->pixel_rows.begin(),
                              thrust::plus<Pixel>() );
        }

        thrust::transform(p_magnitude->pixel_rows.begin(), p_magnitude->pixel_rows.end(),
                          p_magnitude->pixel_rows.begin(),
                          SquareRootOperation<Pixel>() );

        // p = p./max(1,[norm_p; norm_p]);
        thrust::transform(p_magnitude->pixel_rows.begin(), p_magnitude->pixel_rows.end(),
                          p_magnitude->pixel_rows.begin(),
                          MaxOperation<Pixel>(1.0) );

        thrust::transform(p_x_temp->pixel_rows.begin(), p_x_temp->pixel_rows.end(),
                          p_magnitude->pixel_rows.begin(), p_x->pixel_rows.begin(),
                          thrust::divides<Pixel>() );
        thrust::transform(p_y_temp->pixel_rows.begin(), p_y_temp->pixel_rows.end(),
                          p_magnitude->pixel_rows.begin(), p_y->pixel_rows.begin(),
                          thrust::divides<Pixel>() );

        if(u->depth > 1) {
            thrust::transform(p_z_temp->pixel_rows.begin(), p_z_temp->pixel_rows.end(),
                              p_magnitude->pixel_rows.begin(), p_z->pixel_rows.begin(),
                              thrust::divides<Pixel>() );
        }

        // u = u_old - tau * nabla_t * p;
        ThrustThrustImage::divergence(p_x, p_y, p_z,
                                      p_x_temp, p_y_temp, p_z_temp,
                                      divergence_p);

        divergence_p->scale(-tau, divergence_p);
        u_previous->add(divergence_p, u);

        thrust::transform(u->pixel_rows.begin(), u->pixel_rows.end(),
                          f->pixel_rows.begin(), u->pixel_rows.begin(),
                          L1DataTermOperation<Pixel>(tau*lambda) );

        // u_bar = u + theta*(u - u_old);
        thrust::transform(u->pixel_rows.begin(), u->pixel_rows.end(),
                          u_previous->pixel_rows.begin(), u_bar->pixel_rows.begin(),
                          thrust::minus<Pixel>());

        thrust::transform(u_bar->pixel_rows.begin(), u_bar->pixel_rows.end(),
                          u->pixel_rows.begin(), u_bar->pixel_rows.begin(),
                          MultiplyByConstantAndAddOperation<Pixel>(theta) );

        printf("TVL2, iteration=%d / %d \n", iteration_index, iteration_count);

        if(iteration_finished_callback != nullptr &&
           paint_iteration_interval > 0 &&
           iteration_index % paint_iteration_interval == 0 )
            iteration_finished_callback(iteration_index, iteration_count, u);
    }

    delete p_x_temp;
    delete p_y_temp;
    delete p_x;
    delete p_y;

    if(u->depth > 1) {
        delete p_z_temp;
        delete p_z;
    }

    delete divergence_p;
    delete u_previous;
    delete u_bar;

    delete p_magnitude;

    return u;
}



ThrustImage<DevicePixelVector>* filterGPU(ThrustImage<DevicePixelVector>* f,
                                          const Pixel lambda, const unsigned int iteration_count,
                                          const unsigned int paint_iteration_interval,
                                          std::function<void(uint iteration_index, uint iteration_count, ThrustImage<DevicePixelVector>*)>
                                          iteration_finished_callback)
{
    return filter(f, lambda, iteration_count, paint_iteration_interval, iteration_finished_callback);
}

ThrustImage<HostPixelVector>* filterCPU(ThrustImage<HostPixelVector>* f,
                                        const Pixel lambda, const unsigned int iteration_count,
                                        const unsigned int paint_iteration_interval,
                                        std::function<void(uint iteration_index, uint iteration_count, ThrustImage<HostPixelVector>*)> iteration_finished_callback)
{
    return filter(f, lambda, iteration_count, paint_iteration_interval, iteration_finished_callback);
}





