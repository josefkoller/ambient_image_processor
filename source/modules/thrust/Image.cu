#include "Image.cuh"

#include "thrust_operators.cuh"

//#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_TBB


// IMAGE FUNCTIONS...

template<typename PixelVector>
__host__ __device__
Image<PixelVector>::Image(uint width, uint height) : pixel_rows(PixelVector(width*height)),
    width(width), height(height), pixel_count(width*height)
{
}

template<typename PixelVector>
__host__ __device__
void Image<PixelVector>::setPixel(uint x, uint y, Pixel pixel)
{
    uint index = this->getIndex(x,y);
    this->pixel_rows[index] = pixel;
}

template<typename PixelVector>
__host__ __device__
uint Image<PixelVector>::getIndex(uint x, uint y)
{
    return x + y * this->width;
}

template<typename PixelVector>
__host__ __device__
Pixel Image<PixelVector>::getPixel(uint x, uint y)
{
    uint index = this->getIndex(x,y);
    return this->pixel_rows[index];
}


template<typename PixelVector>
__host__ __device__
void Image<PixelVector>::backward_difference_x(Image<PixelVector>* gradient_x)
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
void Image<PixelVector>::forward_difference_x(Image<PixelVector>* gradient_x)
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
void Image<PixelVector>::backward_difference_y(Image<PixelVector>* gradient_y)
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
void Image<PixelVector>::forward_difference_y(Image<PixelVector>* gradient_y)
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
void Image<PixelVector>::laplace(Image<PixelVector>* output_image)
{
    Image<PixelVector>* gradient_x = this->clone_uninitialized();
    Image<PixelVector>* gradient_y = this->clone_uninitialized();
    Image<PixelVector>* gradient_x_back = this->clone_uninitialized();
    Image<PixelVector>* gradient_y_back = this->clone_uninitialized();

    this->forward_difference_x(gradient_x);
    this->forward_difference_y(gradient_y);

    divergence(gradient_x, gradient_y, gradient_x_back, gradient_y_back, output_image);

    delete gradient_x_back;
    delete gradient_y_back;
    delete gradient_x;
    delete gradient_y;

    // matlab: laplace_f = -( nabla_t * (nabla * noise_image) );
    output_image->scale(-1, output_image);
}

template<typename PixelVector>
__host__ __device__
void Image<PixelVector>::square(Image<PixelVector>* squared_image)
{
    thrust::transform(this->pixel_rows.begin(), this->pixel_rows.end(),
                      squared_image->pixel_rows.begin(), SquareOperation<Pixel>());
}

template<typename PixelVector>
__host__ __device__
void Image<PixelVector>::square_root(Image<PixelVector>* square_root_image)
{
    thrust::transform(this->pixel_rows.begin(), this->pixel_rows.end(),
                      square_root_image->pixel_rows.begin(), SquareRootOperation<Pixel>());
}

template<typename PixelVector>
__host__ __device__
void Image<PixelVector>::scale(const Pixel constant_factor, Image<PixelVector>* scaled_image)
{
    thrust::transform(this->pixel_rows.begin(), this->pixel_rows.end(),
                      scaled_image->pixel_rows.begin(), MultiplyByConstant<Pixel>(constant_factor));
}

template<typename PixelVector>
__host__ __device__
void Image<PixelVector>::add(Image<PixelVector>* other, Image<PixelVector>* output)
{
    thrust::transform(this->pixel_rows.begin(), this->pixel_rows.end(),
                      other->pixel_rows.begin(), output->pixel_rows.begin(),
                      thrust::plus<Pixel>());
}

template<typename PixelVector>
__host__ __device__
void Image<PixelVector>::projected_gradient(Image<PixelVector>* gradient_x,
                                            Image<PixelVector>* gradient_y,
                                            Image<PixelVector>* projected_gradient_x,
                                            Image<PixelVector>* projected_gradient_y)
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
void Image<PixelVector>::divergence(Image<PixelVector>* gradient_x, Image<PixelVector>* gradient_y,
                       Image<PixelVector>* gradient_xx, Image<PixelVector>* gradient_yy,
                       Image<PixelVector>* output)
{
    gradient_x->backward_difference_x(gradient_xx);
    gradient_y->backward_difference_y(gradient_yy);

    gradient_xx->add(gradient_yy, output);
}


template<typename PixelVector>
__host__ __device__
Image<PixelVector>* Image<PixelVector>::clone_uninitialized()
{
    return new Image(this->width, this->height);
}


template<typename PixelVector>
__host__ __device__
Image<PixelVector>* Image<PixelVector>::clone_initialized(const Pixel initial_constant_value)
{
    Image<PixelVector>* image = new Image(this->width, this->height);
    thrust::fill(image->pixel_rows.begin(), image->pixel_rows.end(), initial_constant_value);
    return image;
}


template<typename PixelVector>
__host__ __device__
Image<PixelVector>* Image<PixelVector>::clone()
{
    Image<PixelVector>* clone = new Image(this->width, this->height);
    thrust::copy(this->pixel_rows.begin(), this->pixel_rows.end(), clone->pixel_rows.begin());
    return clone;
}


template<typename PixelVector>
__host__ __device__
void Image<PixelVector>::set_pixel_data_of(Image<PixelVector>* image)
{
    thrust::copy(image->pixel_rows.begin(), image->pixel_rows.end(), this->pixel_rows.begin());
}

// explicitly instantiate the template for DevicePixelVector
template DeviceImage::Image(uint width, uint height);
template void DeviceImage::setPixel(uint x, uint y, Pixel pixel);
template Pixel DeviceImage::getPixel(uint x, uint y);
template void DeviceImage::add(DeviceImage* other,
                               DeviceImage* output);
template void DeviceImage::scale(const Pixel constant_factor,
                               DeviceImage* scaled_image);
template void DeviceImage::divergence(DeviceImage* gradient_x, DeviceImage* gradient_y,
                       DeviceImage* gradient_xx, DeviceImage* gradient_yy,
                       DeviceImage* output);
template void DeviceImage::projected_gradient(
    DeviceImage* gradient_x,
    DeviceImage* gradient_y,
    DeviceImage* projected_gradient_x,
    DeviceImage* projected_gradient_y);
template void DeviceImage::backward_difference_x(DeviceImage* gradient_x);
template void DeviceImage::forward_difference_x(DeviceImage* gradient_x);
template void DeviceImage::backward_difference_y(DeviceImage* gradient_y);
template void DeviceImage::forward_difference_y(DeviceImage* gradient_y);
template void DeviceImage::set_pixel_data_of(DeviceImage* image);
template DeviceImage* DeviceImage::clone_uninitialized();
template DeviceImage* DeviceImage::clone_initialized(const Pixel initial_constant_value);
template DeviceImage* DeviceImage::clone();

// explicitly instantiate the template for HostPixelVector
template HostImage::Image(uint width, uint height);
template void HostImage::setPixel(uint x, uint y, Pixel pixel);
template Pixel HostImage::getPixel(uint x, uint y);
template void HostImage::add(HostImage* other,
                               HostImage* output);
template void HostImage::scale(const Pixel constant_factor,
                               HostImage* scaled_image);
template void HostImage::divergence(HostImage* gradient_x, HostImage* gradient_y,
                       HostImage* gradient_xx, HostImage* gradient_yy,
                       HostImage* output);
template void HostImage::projected_gradient(
    HostImage* gradient_x,
    HostImage* gradient_y,
    HostImage* projected_gradient_x,
    HostImage* projected_gradient_y);
template void HostImage::backward_difference_x(HostImage* gradient_x);
template void HostImage::forward_difference_x(HostImage* gradient_x);
template void HostImage::backward_difference_y(HostImage* gradient_y);
template void HostImage::forward_difference_y(HostImage* gradient_y);
template void HostImage::set_pixel_data_of(HostImage* image);
template HostImage* HostImage::clone_uninitialized();
template HostImage* HostImage::clone_initialized(const Pixel initial_constant_value);
template HostImage* HostImage::clone();
