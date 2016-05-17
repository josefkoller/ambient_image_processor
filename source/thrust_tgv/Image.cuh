#ifndef IMAGE_H
#define IMAGE_H



#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CUDA

//#define USE_OPEN_MP_FOR_LOOPS
#ifdef USE_OPEN_MP_FOR_LOOPS
    #define NUMBER_OF_THREADS 4
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
typedef double Pixel;

 __host__ __device__
Pixel max_pixel(Pixel pixel1, Pixel pixel2);

template<typename PixelVector>
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
  void backward_difference_x(Image<PixelVector>* gradient_x);
  __host__ __device__
  void forward_difference_x(Image<PixelVector>* gradient_x);
  __host__ __device__
  void backward_difference_y(Image<PixelVector>* gradient_y);
  __host__ __device__
  void forward_difference_y(Image<PixelVector>* gradient_y);
  __host__ __device__
  void laplace(Image<PixelVector>* output_image);
  __host__ __device__
  void square(Image<PixelVector>* squared_image);
  __host__ __device__
  void square_root(Image<PixelVector>* square_root_image);
  __host__ __device__
  void scale(const Pixel constant_factor, Image<PixelVector>* scaled_image);
  __host__ __device__
  void add(Image<PixelVector>* other, Image<PixelVector>* output);
  __host__ __device__
  static void projected_gradient(Image<PixelVector>* gradient_x, Image<PixelVector>* gradient_y,
                                 Image<PixelVector>* projected_gradient_x, Image<PixelVector>* projected_gradient_y);
  __host__ __device__
  static void divergence(Image<PixelVector>* gradient_x, Image<PixelVector>* gradient_y,
                         Image<PixelVector>* gradient_xx, Image<PixelVector>* gradient_yy,
                         Image<PixelVector>* output);
  __host__ __device__
  Image<PixelVector>* clone_uninitialized();
  __host__ __device__
  Image<PixelVector>* clone_initialized(const Pixel initial_constant_value);
  __host__ __device__
  Image<PixelVector>* clone();

  __host__ __device__
  void set_pixel_data_of(Image<PixelVector>* image);
};

typedef thrust::device_vector<Pixel> DevicePixelVector;
typedef thrust::host_vector<Pixel> HostPixelVector;
typedef Image<DevicePixelVector> DeviceImage;
typedef Image<HostPixelVector> HostImage;


#endif // IMAGE_H
