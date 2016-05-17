
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/adjacent_difference.h>
#include <algorithm>
#include <cuda.h>

#include <stdio.h>

#include "Image.cuh"
#include "thrust_operators.cuh"

template<typename PixelVector>
void print(Image<PixelVector>* image, std::string title)
{
    std::cout << "image: " << title << std::endl;
    for(int y = 0; y < image->height; y++)
    {
        for(int x = 0; x < image->width; x++)
        {
            std::cout << image->getPixel(x,y) << "\t";
        }
        std::cout << std::endl;
    }
}

template<typename PixelVector>
__host__ __device__
Image<PixelVector>* filter(Image<PixelVector>* f,
    const Pixel lambda,
    const unsigned int iteration_count,
    std::function<void(uint iteration_index, uint iteration_count, Image<PixelVector>*)> iteration_finished_callback
     )
{
    typedef Image<PixelVector> ThrustImage;


#if THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_OMP
  omp_set_num_threads(NUMBER_OF_THREADS);
#endif

  printf("TVL2, lambda=%f \n", lambda);

/*
  f = new ThrustImage(3,3);
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

  ThrustImage* u = f->clone();
  ThrustImage* p_x_temp = u->clone_uninitialized(); // memory allocation
  ThrustImage* p_y_temp = u->clone_uninitialized(); // memory allocation
  ThrustImage* p_x = u->clone_initialized(0); // memory allocation
  ThrustImage* p_y = u->clone_initialized(0); // memory allocation
  ThrustImage* p_xx_temp = u->clone_uninitialized(); // memory allocation
  ThrustImage* p_yy_temp = u->clone_uninitialized(); // memory allocation
  ThrustImage* divergence_p = u->clone_uninitialized(); // memory allocation
  ThrustImage* u_bar = u->clone(); // memory allocation
  ThrustImage* u_previous = u->clone_uninitialized(); // memory allocation

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


      u_previous->set_pixel_data_of(u);

      u_bar->forward_difference_x(p_x_temp);
      u_bar->forward_difference_y(p_y_temp);

      thrust::transform(p_x_temp->pixel_rows.begin(), p_x_temp->pixel_rows.end(),
                        p_x->pixel_rows.begin(), p_x_temp->pixel_rows.begin(),
                        MultiplyByConstantAndAddOperation<Pixel>(sigma) );
      thrust::transform(p_y_temp->pixel_rows.begin(), p_y_temp->pixel_rows.end(),
                        p_y->pixel_rows.begin(), p_y_temp->pixel_rows.begin(),
                        MultiplyByConstantAndAddOperation<Pixel>(sigma) );

      ThrustImage::projected_gradient(p_x_temp, p_y_temp, p_x, p_y);

      ThrustImage::divergence(p_x, p_y, p_xx_temp, p_yy_temp, divergence_p);
      thrust::transform(divergence_p->pixel_rows.begin(), divergence_p->pixel_rows.end(),
                        u->pixel_rows.begin(), u->pixel_rows.begin(),
                        MultiplyByConstantAndAddOperation<Pixel>(-tau) ); // minus goes here

      thrust::transform(f->pixel_rows.begin(), f->pixel_rows.end(),
                        u->pixel_rows.begin(), u->pixel_rows.begin(),
                        MultiplyByConstantAndAddOperation<Pixel>(tau*lambda) );

      u->scale(1/(1 + tau*lambda), u);

      thrust::transform(u->pixel_rows.begin(), u->pixel_rows.end(),
                        u_previous->pixel_rows.begin(), u_bar->pixel_rows.begin(),
                        MultiplyByConstantAndAddOperation<Pixel>(-theta));
      u_bar->add(u, u_bar);




      printf("TVL2, iteration=%d / %d \n", iteration_index, iteration_count);

      if(iteration_finished_callback != nullptr)
          iteration_finished_callback(iteration_index, iteration_count, u);

//      print(u, "u");
  }

  delete p_x_temp;
  delete p_y_temp;
  delete p_x;
  delete p_y;
  delete p_xx_temp;
  delete p_yy_temp;
  delete divergence_p;
  delete u_previous;
  delete u_bar;

  return u;
}



Image<DevicePixelVector>* filterGPU(Image<DevicePixelVector>* f,
    const Pixel lambda, const unsigned int iteration_count,
    std::function<void(uint iteration_index, uint iteration_count, Image<DevicePixelVector>*)>
                                    iteration_finished_callback)
{
    return filter(f, lambda, iteration_count, iteration_finished_callback);
}

Image<HostPixelVector>* filterCPU(Image<HostPixelVector>* f,
    const Pixel lambda, const unsigned int iteration_count,
    std::function<void(uint iteration_index, uint iteration_count, Image<HostPixelVector>*)> iteration_finished_callback)
{
    return filter(f, lambda, iteration_count, iteration_finished_callback);
}





