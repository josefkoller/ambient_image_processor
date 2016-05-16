#include "Image.cuh"

#include "thrust_operators.cuh"

//#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_TBB

 __host__ __device__ Pixel max_pixel(Pixel pixel1, Pixel pixel2)
{
  return pixel1 > pixel2 ? pixel1 : pixel2;
}


// IMAGE FUNCTIONS...

__host__ __device__
Image::Image(uint width, uint height) : pixel_rows(PixelVector(width*height)),
    width(width), height(height), pixel_count(width*height)
{
}
__host__ __device__
void Image::setPixel(uint x, uint y, Pixel pixel)
{
    uint index = this->getIndex(x,y);
    this->pixel_rows[index] = pixel;
}
__host__ __device__
uint Image::getIndex(uint x, uint y)
{
    return x + y * this->width;
}
__host__ __device__
Pixel Image::getPixel(uint x, uint y)
{
    uint index = this->getIndex(x,y);
    return this->pixel_rows[index];
}

__host__ __device__
void Image::backward_difference_x(Image* gradient_x)
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
__host__ __device__
void Image::forward_difference_x(Image* gradient_x)
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
__host__ __device__
void Image::backward_difference_y(Image* gradient_y)
{
    PixelVector& data = this->pixel_rows;
    PixelVector& gradient_data = gradient_y->pixel_rows;
    const uint width = this->width;


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

            strided_range<PixelVector::iterator> strided_data(begin, end, width);
            strided_range<PixelVector::iterator> strided_gradient_data(target_begin,
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
__host__ __device__
void Image::forward_difference_y(Image* gradient_y)
{
    PixelVector& data = this->pixel_rows;
    PixelVector& gradient_data = gradient_y->pixel_rows;
    const uint width = this->width;

    PixelVector::iterator::difference_type striding = width;

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


            strided_range<PixelVector::reverse_iterator> strided_data(begin, end, striding);
            strided_range<PixelVector::reverse_iterator> strided_gradient_data(target_begin,
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
__host__ __device__
void Image::laplace(Image* output_image)
{
    Image* gradient_x = this->clone_uninitialized();
    Image* gradient_y = this->clone_uninitialized();
    Image* gradient_x_back = this->clone_uninitialized();
    Image* gradient_y_back = this->clone_uninitialized();

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
__host__ __device__
void Image::square(Image* squared_image)
{
    thrust::transform(this->pixel_rows.begin(), this->pixel_rows.end(),
                      squared_image->pixel_rows.begin(), SquareOperation<Pixel>());
}
__host__ __device__
void Image::square_root(Image* square_root_image)
{
    thrust::transform(this->pixel_rows.begin(), this->pixel_rows.end(),
                      square_root_image->pixel_rows.begin(), SquareRootOperation<Pixel>());
}
__host__ __device__
void Image::scale(const Pixel constant_factor, Image* scaled_image)
{
    thrust::transform(this->pixel_rows.begin(), this->pixel_rows.end(),
                      scaled_image->pixel_rows.begin(), MultiplyByConstant<Pixel>(constant_factor));
}
__host__ __device__
void Image::add(Image* other, Image* output)
{
    thrust::transform(this->pixel_rows.begin(), this->pixel_rows.end(),
                      other->pixel_rows.begin(), output->pixel_rows.begin(),
                      thrust::plus<Pixel>());
}
__host__ __device__
void Image::projected_gradient(Image* gradient_x, Image* gradient_y,
                               Image* projected_gradient_x, Image* projected_gradient_y)
{
    thrust::transform(gradient_x->pixel_rows.begin(), gradient_x->pixel_rows.end(),
                      gradient_y->pixel_rows.begin(), projected_gradient_x->pixel_rows.begin(),
                      ProjectNormalizedGradientMagnitude1<Pixel>());
    thrust::transform(gradient_x->pixel_rows.begin(), gradient_x->pixel_rows.end(),
                      gradient_y->pixel_rows.begin(), projected_gradient_y->pixel_rows.begin(),
                      ProjectNormalizedGradientMagnitude2<Pixel>());
}
__host__ __device__
void Image::divergence(Image* gradient_x, Image* gradient_y,
                       Image* gradient_xx, Image* gradient_yy, Image* output)
{
    gradient_x->backward_difference_x(gradient_xx);
    gradient_y->backward_difference_y(gradient_yy);

    gradient_xx->add(gradient_yy, output);
}

__host__ __device__
Image* Image::clone_uninitialized()
{
    return new Image(this->width, this->height);
}

__host__ __device__
Image* Image::clone_initialized(const Pixel initial_constant_value)
{
    Image* image = new Image(this->width, this->height);
    thrust::fill(image->pixel_rows.begin(), image->pixel_rows.end(), initial_constant_value);
    return image;
}
