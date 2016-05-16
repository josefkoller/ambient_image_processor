
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/adjacent_difference.h>
#include <algorithm>
#include <cuda.h>

#include "Image.cuh"
#include "thrust_operators.cuh"

__host__ __device__
Image* filter(Image* f,
    const float lambda,
    const unsigned int iteration_count)
{
#if THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_OMP
  omp_set_num_threads(NUMBER_OF_THREADS);
#endif

  const float sqrt_8 = std::sqrt(8.0f); // algorithm paramteter
  float tau = 1.0f / sqrt_8;
  float sigma = tau;
//  const float gamma = 0.7f * lambda;  // algorithm paramteter
  float theta = 1.0f; // will be used later

  /*
  Image* gradient_x_f = f->clone_uninitialized();
  f->forward_difference_x(gradient_x_f);
  Image* gradient_y_f = f->clone_uninitialized();
  f->forward_difference_y(gradient_y_f);
  */

  Image* u = f->clone();
  Image* p_x_temp = u->clone_uninitialized(); // memory allocation
  Image* p_y_temp = u->clone_uninitialized(); // memory allocation
  Image* p_x = u->clone_initialized(0); // memory allocation
  Image* p_y = u->clone_initialized(0); // memory allocation
  Image* p_xx_temp = u->clone_uninitialized(); // memory allocation
  Image* p_yy_temp = u->clone_uninitialized(); // memory allocation
  Image* divergence_p = u->clone_uninitialized(); // memory allocation
  Image* u_bar = u->clone(); // memory allocation
  Image* u_previous = u->clone_uninitialized(); // memory allocation

  Image* gradient_magnitude_u = u->clone_uninitialized();
  Image* gradient_x_difference = u->clone_uninitialized();
  Image* gradient_y_difference = u->clone_uninitialized();
  Image* gradient_difference_magnitude = u->clone_uninitialized();

  f->scale(tau * lambda, f); // for later

  for(uint iteration_index = 0; iteration_index < iteration_count; iteration_index++)
  {
      u_previous->setPixelDataOf(u);

      u_bar->forward_difference_x(p_x_temp);
      u_bar->forward_difference_y(p_y_temp);

      // begin computing energy...
      /* energy, matlab:
       temp_p = nabla * u;
       gap = sum(sqrt(temp_p(1:N).^2 + temp_p(N+1:2*N).^2)) +...
           sum((nabla * u - nabla * f).^2) * lambda/2;
       */

      /*
      thrust::transform(p_x_temp->pixel_rows.begin(), p_x_temp->pixel_rows.end(),
                        p_y_temp->pixel_rows.begin(), gradient_magnitude_u->pixel_rows.begin(),
                        GradientMagnitude<Pixel>());
      float energy1 = thrust::reduce(gradient_magnitude_u->pixel_rows.begin(),
                               gradient_magnitude_u->pixel_rows.end(),
                               0, // init
                               thrust::plus<Pixel>() );

      thrust::transform(p_x_temp->pixel_rows.begin(), p_x_temp->pixel_rows.end(),
                        gradient_x_f->pixel_rows.begin(),
                        gradient_x_difference->pixel_rows.begin(),
                        thrust::minus<Pixel>());
      thrust::transform(p_y_temp->pixel_rows.begin(), p_y_temp->pixel_rows.end(),
                        gradient_y_f->pixel_rows.begin(),
                        gradient_y_difference->pixel_rows.begin(),
                        thrust::minus<Pixel>());
      thrust::transform(gradient_x_difference->pixel_rows.begin(), gradient_x_difference->pixel_rows.end(),
                        gradient_y_difference->pixel_rows.begin(), gradient_difference_magnitude->pixel_rows.begin(),
                        GradientMagnitudeSquare<Pixel>());
      std::cout << "energy1: " << energy1 << std::endl;
      float energy = energy1 + lambda*0.5f * thrust::reduce(
                  gradient_difference_magnitude->pixel_rows.begin(),
                  gradient_difference_magnitude->pixel_rows.end(),
                  0, // init
                  thrust::plus<Pixel>());
      std::cout << "energy: " << energy << std::endl;
      // end energy
      */

      // p update ... matlab: temp_p  = nabla * head_u * sigma + p;
      thrust::transform(p_x_temp->pixel_rows.begin(), p_x_temp->pixel_rows.end(),
                        p_x->pixel_rows.begin(), p_x_temp->pixel_rows.begin(),
                        MultiplyByConstantAndAddOperation<Pixel>(sigma) );
      thrust::transform(p_y_temp->pixel_rows.begin(), p_y_temp->pixel_rows.end(),
                        p_y->pixel_rows.begin(), p_y_temp->pixel_rows.begin(),
                        MultiplyByConstantAndAddOperation<Pixel>(sigma) );// JJ reuse operation?
      // project/normalize p
      Image::projected_gradient(p_x_temp, p_y_temp, p_x, p_y);

      // u update1...  matlab: u = old_u + tau * lambda * (-nabla_t * p * (1/lambda + 1)+ f);
      Image::divergence(p_x, p_y, p_xx_temp, p_yy_temp, divergence_p);

      thrust::transform(divergence_p->pixel_rows.begin(), divergence_p->pixel_rows.end(),
                        u->pixel_rows.begin(), u->pixel_rows.begin(),
                        MultiplyByConstantAndAddOperation<Pixel>(-1) ); // minus goes here

      u->add(f, u);
      u->scale(1/(1 + tau*lambda), u);

      thrust::transform(u->pixel_rows.begin(), u->pixel_rows.end(),
                        u_previous->pixel_rows.begin(), u_bar->pixel_rows.begin(),
                        MultiplyByConstantAndAddOperation<Pixel>(-1));
      u_bar->scale(theta, u_bar);
      u_bar->add(u, u_bar);

      std::cout << "iteration: " << (iteration_index+1) << "/" << iteration_count << std::endl;
  }
  delete gradient_difference_magnitude;
  delete gradient_x_difference;
  delete gradient_y_difference;
//  delete gradient_x_f;
//  delete gradient_y_f;
  delete gradient_magnitude_u;

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


