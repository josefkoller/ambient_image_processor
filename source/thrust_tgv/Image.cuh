#ifndef IMAGE_H
#define IMAGE_H


#define NUMBER_OF_THREADS 4
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CUDA
// #define USE_OPEN_MP_FOR_LOOPS
#ifdef USE_OPEN_MP_FOR_LOOPS
    #include <omp.h>
#endif
//#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_OMP
#if THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_OMP
    #include <omp.h>
#endif

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/adjacent_difference.h>
#include <algorithm>

typedef unsigned int uint;
typedef float Pixel;
//typedef thrust::device_vector<Pixel> PixelVector;
typedef thrust::host_vector<Pixel> PixelVector;

 __host__ __device__
Pixel max_pixel(Pixel pixel1, Pixel pixel2);

struct Image
{
  PixelVector pixel_rows;
  const uint width;
  const uint height;
  const uint pixel_count;
  __host__ __device__
  Image(uint width, uint height);
  __host__ __device__
  void setPixel(uint x, uint y, Pixel pixel);
  __host__ __device__
  uint getIndex(uint x, uint y);
  __host__ __device__
  Pixel getPixel(uint x, uint y);
  __host__ __device__
  void backward_difference_x(Image* gradient_x);
  __host__ __device__
  void forward_difference_x(Image* gradient_x);
  __host__ __device__
  void backward_difference_y(Image* gradient_y);
  __host__ __device__
  void forward_difference_y(Image* gradient_y);
  __host__ __device__
  void laplace(Image* output_image);
  __host__ __device__
  void square(Image* squared_image);
  __host__ __device__
  void square_root(Image* square_root_image);
  __host__ __device__
  void scale(const Pixel constant_factor, Image* scaled_image);
  __host__ __device__
  void add(Image* other, Image* output);
  __host__ __device__
  static void projected_gradient(Image* gradient_x, Image* gradient_y,
                                 Image* projected_gradient_x, Image* projected_gradient_y);
  __host__ __device__
  static void divergence(Image* gradient_x, Image* gradient_y,
                         Image* gradient_xx, Image* gradient_yy, Image* output);
  __host__ __device__
  Image* clone_uninitialized();
  __host__ __device__
  Image* clone_initialized(const Pixel initial_constant_value);
  __host__ __device__
  Image* clone();

  __host__ __device__
  void setPixelDataOf(Image* image);
};

#endif // IMAGE_H
