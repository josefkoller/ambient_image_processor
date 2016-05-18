#include "ThrustImage.cuh"

#include "thrust_operators.cuh"

//#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_TBB


#pragma hd_warning_disable // calling a host function in a device...

template<typename PixelVector>
__host__ __device__
ThrustImage<PixelVector>::ThrustImage(uint width, uint height) : pixel_rows(PixelVector(width*height)),
    width(width), height(height), pixel_count(width*height)
{
}

template<typename PixelVector>
__host__ __device__
void ThrustImage<PixelVector>::setPixel(uint x, uint y, Pixel pixel)
{
    uint index = this->getIndex(x,y);
    this->pixel_rows[index] = pixel;
}

template<typename PixelVector>
__host__ __device__
uint ThrustImage<PixelVector>::getIndex(uint x, uint y)
{
    return x + y * this->width;
}

template<typename PixelVector>
__host__ __device__
Pixel ThrustImage<PixelVector>::getPixel(uint x, uint y)
{
    uint index = this->getIndex(x,y);
    return this->pixel_rows[index];
}


template<typename PixelVector>
__host__ __device__
void ThrustImage<PixelVector>::backward_difference_x(ThrustImage<PixelVector>* gradient_x)
{
  InverseMinus<Pixel> inverse_minus;

  PixelVector& data = this->pixel_rows;
  PixelVector& gradient_data = gradient_x->pixel_rows;
  const uint width = this->width;
  const uint height = this->height;

#ifdef USE_OPEN_MP_FOR_LOOPS
  omp_set_num_threads(NUMBER_OF_THREADS);
  #pragma omp parallel shared(data,gradient_data)
  {
      #pragma omp for
#endif
      for(uint row_index = 0; row_index < height; row_index++)
      {
          auto offset = row_index * width;
          auto begin = data.begin() + offset;
          auto end = begin + this->width;
          auto target_begin = gradient_data.begin() + offset;
          thrust::adjacent_difference(begin, end, target_begin, inverse_minus);

          // neumann boundary conditions for u means...
          // the value of p at the boundary is constant
          // this is done by thrust::adjacent_difference, but
          // the funny sign convention has to be fullfilled
          *target_begin = -(*target_begin);
      }
#ifdef USE_OPEN_MP_FOR_LOOPS
  }
#endif
}

template<typename PixelVector>
__host__ __device__
void ThrustImage<PixelVector>::forward_difference_x(ThrustImage<PixelVector>* gradient_x)
{
    PixelVector& data = this->pixel_rows;
    PixelVector& gradient_data = gradient_x->pixel_rows;
    const uint width = this->width;
    const uint height = this->height;

#ifdef USE_OPEN_MP_FOR_LOOPS
    omp_set_num_threads(NUMBER_OF_THREADS);
    #pragma omp parallel shared(data,gradient_data)
    {
        #pragma omp for
#endif
        for(uint row_index = 0; row_index < height; row_index++)
        {
            auto offset = row_index * width;
            auto begin = data.rbegin() + offset;
            auto end = begin + width;
            auto target_begin = gradient_data.rbegin() + offset;
            thrust::adjacent_difference(begin, end, target_begin, InverseMinus<Pixel>());

            // neumann boundary conditions: 0 in the last column...
            *target_begin = 0;
        }
#ifdef USE_OPEN_MP_FOR_LOOPS
    }
#endif
}

template<typename PixelVector>
__host__ __device__
void ThrustImage<PixelVector>::backward_difference_y(ThrustImage<PixelVector>* gradient_y)
{
    PixelVector& data = this->pixel_rows;
    PixelVector& gradient_data = gradient_y->pixel_rows;
    const uint width = this->width;

    typedef typename PixelVector::iterator iterator;

#ifdef USE_OPEN_MP_FOR_LOOPS
    omp_set_num_threads(NUMBER_OF_THREADS);
    #pragma omp parallel shared(data,gradient_data)
    {
        #pragma omp for
#endif
        for(uint column_index = 0; column_index < width; column_index++)
        {
            auto begin = data.begin() + column_index;
            auto end = data.end() + column_index;
            auto target_begin = gradient_data.begin() + column_index;
            auto target_end = gradient_data.end() + column_index;

            strided_range<iterator> strided_data(begin, end, width);
            strided_range<iterator> strided_gradient_data(target_begin,
                                                                       target_end, width);

            auto column_begin = strided_data.begin();
            auto column_end = strided_data.end();
            auto column_target_begin = strided_gradient_data.begin();

            thrust::adjacent_difference(column_begin, column_end, column_target_begin,
                                        InverseMinus<Pixel>());

            // neumann boundary conditions for u means...
            // the value of p at the boundary is constant
            // this is done by thrust::adjacent_difference, but
            // the funny sign convention has to be fullfilled
            auto target_begin2 = gradient_data.begin() + column_index;
            *target_begin2 = -(*target_begin2);
        }
#ifdef USE_OPEN_MP_FOR_LOOPS
    }
#endif
}

template<typename PixelVector>
__host__ __device__
void ThrustImage<PixelVector>::forward_difference_y(ThrustImage<PixelVector>* gradient_y)
{
    PixelVector& data = this->pixel_rows;
    PixelVector& gradient_data = gradient_y->pixel_rows;
    const uint width = this->width;

    typedef typename PixelVector::iterator::difference_type difference_type;
    difference_type striding = width;

    typedef typename PixelVector::reverse_iterator reverse_iterator;

#ifdef USE_OPEN_MP_FOR_LOOPS
    omp_set_num_threads(NUMBER_OF_THREADS);
    #pragma omp parallel shared(data,gradient_data)
    {
        #pragma omp for
#endif
        for(uint column_index = 0; column_index < width; column_index++)
        {
            auto begin = data.rbegin() + column_index;
            auto end = data.rend() + column_index;
            auto target_begin = gradient_data.rbegin() + column_index;
            auto target_end = gradient_data.rend() + column_index;


            strided_range<reverse_iterator> strided_data(begin, end, striding);
            strided_range<reverse_iterator> strided_gradient_data(target_begin,
                                                                       target_end, width);

            auto column_begin = strided_data.begin();
            auto column_end = strided_data.end();
            auto column_target_begin = strided_gradient_data.begin();

            thrust::adjacent_difference(column_begin, column_end,
                                        column_target_begin,
                                        InverseMinus<Pixel>());

            // neumann boundary conditions: 0 in the first row...
            auto target_begin2 = gradient_data.rbegin() + column_index;
            *target_begin2 = 0;
        }
#ifdef USE_OPEN_MP_FOR_LOOPS
    }
#endif
}

template<typename PixelVector>
__host__ __device__
void ThrustImage<PixelVector>::laplace(ThrustImage<PixelVector>* output_ThrustImage)
{
    ThrustImage<PixelVector>* gradient_x = this->clone_uninitialized();
    ThrustImage<PixelVector>* gradient_y = this->clone_uninitialized();
    ThrustImage<PixelVector>* gradient_x_back = this->clone_uninitialized();
    ThrustImage<PixelVector>* gradient_y_back = this->clone_uninitialized();

    this->forward_difference_x(gradient_x);
    this->forward_difference_y(gradient_y);

    divergence(gradient_x, gradient_y, gradient_x_back, gradient_y_back, output_ThrustImage);

    delete gradient_x_back;
    delete gradient_y_back;
    delete gradient_x;
    delete gradient_y;

    // matlab: laplace_f = -( nabla_t * (nabla * noise_ThrustImage) );
    output_ThrustImage->scale(-1, output_ThrustImage);
}

template<typename PixelVector>
__host__ __device__
void ThrustImage<PixelVector>::square(ThrustImage<PixelVector>* squared_ThrustImage)
{
    thrust::transform(this->pixel_rows.begin(), this->pixel_rows.end(),
                      squared_ThrustImage->pixel_rows.begin(), SquareOperation<Pixel>());
}

template<typename PixelVector>
__host__ __device__
void ThrustImage<PixelVector>::square_root(ThrustImage<PixelVector>* square_root_ThrustImage)
{
    thrust::transform(this->pixel_rows.begin(), this->pixel_rows.end(),
                      square_root_ThrustImage->pixel_rows.begin(), SquareRootOperation<Pixel>());
}

template<typename PixelVector>
__host__ __device__
void ThrustImage<PixelVector>::scale(const Pixel constant_factor, ThrustImage<PixelVector>* scaled_ThrustImage)
{
    thrust::transform(this->pixel_rows.begin(), this->pixel_rows.end(),
                      scaled_ThrustImage->pixel_rows.begin(), MultiplyByConstant<Pixel>(constant_factor));
}

template<typename PixelVector>
__host__ __device__
void ThrustImage<PixelVector>::add(ThrustImage<PixelVector>* other, ThrustImage<PixelVector>* output)
{
    thrust::transform(this->pixel_rows.begin(), this->pixel_rows.end(),
                      other->pixel_rows.begin(), output->pixel_rows.begin(),
                      thrust::plus<Pixel>());
}

template<typename PixelVector>
__host__ __device__
void ThrustImage<PixelVector>::projected_gradient(ThrustImage<PixelVector>* gradient_x,
                                            ThrustImage<PixelVector>* gradient_y,
                                            ThrustImage<PixelVector>* projected_gradient_x,
                                            ThrustImage<PixelVector>* projected_gradient_y)
{
    thrust::transform(gradient_x->pixel_rows.begin(), gradient_x->pixel_rows.end(),
                      gradient_y->pixel_rows.begin(), projected_gradient_x->pixel_rows.begin(),
                      ProjectNormalizedGradientMagnitude1<Pixel>());

    thrust::transform(gradient_y->pixel_rows.begin(), gradient_y->pixel_rows.end(),
                      gradient_x->pixel_rows.begin(), projected_gradient_y->pixel_rows.begin(),
                      ProjectNormalizedGradientMagnitude1<Pixel>());
    /*

    thrust::transform(gradient_x->pixel_rows.begin(), gradient_x->pixel_rows.end(),
                      gradient_y->pixel_rows.begin(), projected_gradient_y->pixel_rows.begin(),
                      ProjectNormalizedGradientMagnitude2<Pixel>());
                      */
}

template<typename PixelVector>
__host__ __device__
void ThrustImage<PixelVector>::divergence(ThrustImage<PixelVector>* gradient_x, ThrustImage<PixelVector>* gradient_y,
                       ThrustImage<PixelVector>* gradient_xx, ThrustImage<PixelVector>* gradient_yy,
                       ThrustImage<PixelVector>* output)
{
    gradient_x->backward_difference_x(gradient_xx);
    gradient_y->backward_difference_y(gradient_yy);

    gradient_xx->add(gradient_yy, output);
}


template<typename PixelVector>
__host__ __device__
ThrustImage<PixelVector>* ThrustImage<PixelVector>::clone_uninitialized()
{
    return new ThrustImage(this->width, this->height);
}


template<typename PixelVector>
__host__ __device__
ThrustImage<PixelVector>* ThrustImage<PixelVector>::clone_initialized(const Pixel initial_constant_value)
{
    ThrustImage<PixelVector>* image = new ThrustImage(this->width, this->height);
    thrust::fill(image->pixel_rows.begin(), image->pixel_rows.end(), initial_constant_value);
    return image;
}


template<typename PixelVector>
__host__ __device__
ThrustImage<PixelVector>* ThrustImage<PixelVector>::clone()
{
    ThrustImage<PixelVector>* clone = new ThrustImage(this->width, this->height);
    thrust::copy(this->pixel_rows.begin(), this->pixel_rows.end(), clone->pixel_rows.begin());
    return clone;
}


template<typename PixelVector>
__host__ __device__
void ThrustImage<PixelVector>::set_pixel_data_of(ThrustImage<PixelVector>* ThrustImage)
{
    thrust::copy(ThrustImage->pixel_rows.begin(), ThrustImage->pixel_rows.end(), this->pixel_rows.begin());
}

// explicitly instantiate the template for DevicePixelVector
template DeviceThrustImage::ThrustImage(uint width, uint height);
template void DeviceThrustImage::setPixel(uint x, uint y, Pixel pixel);
template Pixel DeviceThrustImage::getPixel(uint x, uint y);
template void DeviceThrustImage::add(DeviceThrustImage* other,
                               DeviceThrustImage* output);
template void DeviceThrustImage::scale(const Pixel constant_factor,
                               DeviceThrustImage* scaled_ThrustImage);
template void DeviceThrustImage::divergence(DeviceThrustImage* gradient_x, DeviceThrustImage* gradient_y,
                       DeviceThrustImage* gradient_xx, DeviceThrustImage* gradient_yy,
                       DeviceThrustImage* output);
template void DeviceThrustImage::projected_gradient(
    DeviceThrustImage* gradient_x,
    DeviceThrustImage* gradient_y,
    DeviceThrustImage* projected_gradient_x,
    DeviceThrustImage* projected_gradient_y);
template void DeviceThrustImage::backward_difference_x(DeviceThrustImage* gradient_x);
template void DeviceThrustImage::forward_difference_x(DeviceThrustImage* gradient_x);
template void DeviceThrustImage::backward_difference_y(DeviceThrustImage* gradient_y);
template void DeviceThrustImage::forward_difference_y(DeviceThrustImage* gradient_y);
template void DeviceThrustImage::set_pixel_data_of(DeviceThrustImage* ThrustImage);
template DeviceThrustImage* DeviceThrustImage::clone_uninitialized();
template DeviceThrustImage* DeviceThrustImage::clone_initialized(const Pixel initial_constant_value);
template DeviceThrustImage* DeviceThrustImage::clone();

// explicitly instantiate the template for HostPixelVector
template HostThrustImage::ThrustImage(uint width, uint height);
template void HostThrustImage::setPixel(uint x, uint y, Pixel pixel);
template Pixel HostThrustImage::getPixel(uint x, uint y);
template void HostThrustImage::add(HostThrustImage* other,
                               HostThrustImage* output);
template void HostThrustImage::scale(const Pixel constant_factor,
                               HostThrustImage* scaled_ThrustImage);
template void HostThrustImage::divergence(HostThrustImage* gradient_x, HostThrustImage* gradient_y,
                       HostThrustImage* gradient_xx, HostThrustImage* gradient_yy,
                       HostThrustImage* output);
template void HostThrustImage::projected_gradient(
    HostThrustImage* gradient_x,
    HostThrustImage* gradient_y,
    HostThrustImage* projected_gradient_x,
    HostThrustImage* projected_gradient_y);
template void HostThrustImage::backward_difference_x(HostThrustImage* gradient_x);
template void HostThrustImage::forward_difference_x(HostThrustImage* gradient_x);
template void HostThrustImage::backward_difference_y(HostThrustImage* gradient_y);
template void HostThrustImage::forward_difference_y(HostThrustImage* gradient_y);
template void HostThrustImage::set_pixel_data_of(HostThrustImage* ThrustImage);
template HostThrustImage* HostThrustImage::clone_uninitialized();
template HostThrustImage* HostThrustImage::clone_initialized(const Pixel initial_constant_value);
template HostThrustImage* HostThrustImage::clone();
