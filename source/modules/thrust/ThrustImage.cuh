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

template<typename PixelVector>
struct ThrustImage
{
  typedef PixelVector Vector;

  PixelVector pixel_rows;
  const uint width;
  const uint height;
  const uint depth;
  const uint pixel_count;

  __host__ __device__
  ThrustImage(uint width, uint height, uint depth);
  __host__ __device__
  ThrustImage(uint width, uint height, uint depth, Vector pixel_rows);


  __host__ __device__
  void setPixel(uint x, uint y, uint z, Pixel pixel);
  __host__ __device__
  uint getIndex(uint x, uint y, uint z);
  __host__ __device__
  Pixel getPixel(uint x, uint y, uint z);
  __host__ __device__
  void backward_difference_x(ThrustImage<PixelVector>* gradient_x);
  __host__ __device__
  void forward_difference_x(ThrustImage<PixelVector>* gradient_x);
  __host__ __device__
  void backward_difference_y(ThrustImage<PixelVector>* gradient_y);
  __host__ __device__
  void forward_difference_y(ThrustImage<PixelVector>* gradient_y);

  // TODO forward, backward z

  __host__ __device__
  void laplace(ThrustImage<PixelVector>* output_image);
  __host__ __device__
  void square(ThrustImage<PixelVector>* squared_image);
  __host__ __device__
  void square_root(ThrustImage<PixelVector>* square_root_image);
  __host__ __device__
  void scale(const Pixel constant_factor, ThrustImage<PixelVector>* scaled_image);
  __host__ __device__
  void add(ThrustImage<PixelVector>* other, ThrustImage<PixelVector>* output);
  __host__ __device__
  static void projected_gradient(ThrustImage<PixelVector>* gradient_x, ThrustImage<PixelVector>* gradient_y,
                                 ThrustImage<PixelVector>* projected_gradient_x, ThrustImage<PixelVector>* projected_gradient_y);
  __host__ __device__
  static void divergence(ThrustImage<PixelVector>* gradient_x, ThrustImage<PixelVector>* gradient_y,
                         ThrustImage<PixelVector>* gradient_xx, ThrustImage<PixelVector>* gradient_yy,
                         ThrustImage<PixelVector>* output);
  __host__ __device__
  ThrustImage<PixelVector>* clone_uninitialized();
  __host__ __device__
  ThrustImage<PixelVector>* clone_initialized(const Pixel initial_constant_value);
  __host__ __device__
  ThrustImage<PixelVector>* clone();

  __host__ __device__
  void set_pixel_data_of(ThrustImage<PixelVector>* image);
};

typedef thrust::device_vector<Pixel> DevicePixelVector;
typedef thrust::host_vector<Pixel> HostPixelVector;
typedef ThrustImage<DevicePixelVector> DeviceThrustImage;
typedef ThrustImage<HostPixelVector> HostThrustImage;


#endif // IMAGE_H
