#include "ImageVectorOperations.h"

#include <cuda_runtime.h>
#include "ImageMatrix.h"

template<typename Pixel>
void image_multiply_kernel_launch(Pixel* image1, Pixel* image2, Pixel* temp,
                                  ImageVectorOperations::Dimension voxel_count);

template<typename Pixel>
void image_multiply_matrix_kernel_launch(Pixel* matrix_elements, Pixel* vector, Pixel* result,
                                         ImageVectorOperations::Dimension voxel_count);

template<typename Pixel>
void image_scale_kernel_launch(Pixel* image1, Pixel factor, Pixel* result,
                             ImageVectorOperations::Dimension voxel_count);

template<typename Pixel>
void image_subtract_kernel_launch(Pixel* image1, Pixel* image2, Pixel* result,
                             ImageVectorOperations::Dimension voxel_count);

template<typename Pixel>
void image_add_kernel_launch(Pixel* image1, Pixel* image2, Pixel* result,
                             ImageVectorOperations::Dimension voxel_count);

template<typename Pixel>
void image_assign_kernel_launch(Pixel* source, Pixel* destination,
                             ImageVectorOperations::Dimension voxel_count);

template<typename Pixel>
void laplace_kernel_launch(Pixel* source, Pixel* destination,
                           ImageVectorOperations::Dimension image_width,
                           ImageVectorOperations::Dimension image_height,
                           ImageVectorOperations::Dimension image_depth);

template<typename Pixel>
void image_set_zeros_kernel_launch(Pixel* image,
                             ImageVectorOperations::Dimension voxel_count);

ImageVectorOperations::ImageVectorOperations()
{
}


template<typename Pixel>
void ImageVectorOperations::setZeros(Pixel* image, Dimension voxel_count)
{
    image_set_zeros_kernel_launch(image, voxel_count);
}

template<typename Pixel>
Pixel ImageVectorOperations::scalarProduct(Pixel* image1, Pixel* image2, Pixel* temp, Dimension voxel_count)
{
    image_multiply_kernel_launch(image1, image2, temp, voxel_count);

    Pixel result = 0;
    for(int i = 0; i < voxel_count; i++)
        result += temp[i];

    return result;
}

template<typename Pixel>
void ImageVectorOperations::matrixVectorMultiply(ImageMatrix<Pixel>* matrix, Pixel* vector, Pixel* result)
{
    image_multiply_matrix_kernel_launch(matrix->elements, vector, result, matrix->voxel_count);
}

template<typename Pixel>
void ImageVectorOperations::add(Pixel* image1, Pixel* image2, Pixel* result, Dimension voxel_count)
{
    image_add_kernel_launch(image1, image2, result, voxel_count);
}

template<typename Pixel>
void ImageVectorOperations::subtract(Pixel* image1, Pixel* image2, Pixel* result, Dimension voxel_count)
{
    image_subtract_kernel_launch(image1, image2, result, voxel_count);
}

template<typename Pixel>
void ImageVectorOperations::scale(Pixel* image1, Pixel factor, Pixel* result, Dimension voxel_count)
{
    image_scale_kernel_launch(image1, factor, result, voxel_count);
}

template<typename Pixel>
void ImageVectorOperations::assign(Pixel* source, Pixel* destination, Dimension voxel_count)
{
    image_assign_kernel_launch(source, destination, voxel_count);
}

template<typename Pixel>
void ImageVectorOperations::laplace(Pixel* vector, Pixel* result,
  Dimension image_width, Dimension image_height, Dimension image_depth)
{
    laplace_kernel_launch(vector, result, image_width, image_height, image_depth);
}


template void ImageVectorOperations::setZeros(float* vector, Dimension voxel_count);
template void ImageVectorOperations::laplace(float* vector, float* result,
  Dimension image_width, Dimension image_height, Dimension image_depth);
template float ImageVectorOperations::scalarProduct(float* image1, float* image2, float* temp, Dimension voxel_count);
template void ImageVectorOperations::add(float* image1, float* image2, float* result, Dimension voxel_count);
template void ImageVectorOperations::subtract(float* image1, float* image2, float* result, Dimension voxel_count);
template void ImageVectorOperations::scale(float* image1, float factor, float* result, Dimension voxel_count);
template void ImageVectorOperations::assign(float* source, float* destination, Dimension voxel_count);
template void ImageVectorOperations::matrixVectorMultiply(ImageMatrix<float>* matrix, float* vector, float* result);

template void ImageVectorOperations::setZeros(double* vector, Dimension voxel_count);
template void ImageVectorOperations::laplace(double* vector, double* result,
  Dimension image_width, Dimension image_height, Dimension image_depth);
template double ImageVectorOperations::scalarProduct(double* image1, double* image2, double* temp, Dimension voxel_count);
template void ImageVectorOperations::add(double* image1, double* image2, double* result, Dimension voxel_count);
template void ImageVectorOperations::subtract(double* image1, double* image2, double* result, Dimension voxel_count);
template void ImageVectorOperations::scale(double* image1, double factor, double* result, Dimension voxel_count);
template void ImageVectorOperations::assign(double* source, double* destination, Dimension voxel_count);
template void ImageVectorOperations::matrixVectorMultiply(ImageMatrix<double>* matrix, double* vector, double* result);
