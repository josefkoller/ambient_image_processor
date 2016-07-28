#include "ResizeProcessor.h"

#include <itkResampleImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkScaleTransform.h>

#include <itkWindowedSincInterpolateImageFunction.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

ResizeProcessor::ResizeProcessor()
{
}

ITKImage ResizeProcessor::process(ITKImage image,
                                  ITKImage::PixelType size_factor,
                                  ResizeProcessor::InterpolationMethod interpolation_method)
{
    uint width = image.width * size_factor;
    uint height = image.height * size_factor;
    uint depth = image.depth == 1 ? 1 : image.depth * size_factor;

    return process(image, size_factor, width, height, depth, interpolation_method);
}

ITKImage ResizeProcessor::process(ITKImage image, ITKImage::PixelType size_factor,
                                  uint width, uint height, uint depth,
                                  ResizeProcessor::InterpolationMethod interpolation_method)
{
    typedef ITKImage::InnerITKImage Image;
    typedef itk::ResampleImageFilter<Image, Image> ResampleFilter;
    typedef itk::ScaleTransform<Image::PixelType, Image::ImageDimension>  Transform;

    typedef itk::NearestNeighborInterpolateImageFunction<Image> NearestNeighborInterpolation;
    typedef itk::LinearInterpolateImageFunction<Image> LinearInterpolation;
    typedef itk::WindowedSincInterpolateImageFunction<Image, 4> SincInterpolation;

    Transform::Pointer transform = Transform::New();
    Transform::ScaleType scale;
    scale.Fill(1 / size_factor);
    transform->SetScale(scale);

    Image::PointType origin;
    origin.Fill(0);
    Image::SpacingType spacing;
    spacing.Fill(1);
    Image::DirectionType direction;
    direction.SetIdentity();

    Image::SizeType image_size = image.getPointer()->GetLargestPossibleRegion().GetSize();
    Image::SizeType size;
    size[0] = width;
    size[1] = height;
    size[2] = depth;

    std::cout << "resizing from : " << image_size << std::endl;
    std::cout << "to : " << size << std::endl;

    ResampleFilter::Pointer resample_filter = ResampleFilter::New();
    resample_filter->SetDefaultPixelValue(0);
    resample_filter->SetTransform(transform);

    if(interpolation_method == InterpolationMethod::NearestNeighbour)
        resample_filter->SetInterpolator(NearestNeighborInterpolation::New());
    else if(interpolation_method == InterpolationMethod::Linear)
        resample_filter->SetInterpolator(LinearInterpolation::New());
    else
        resample_filter->SetInterpolator(SincInterpolation::New());

    resample_filter->SetSize(size);
    resample_filter->SetInput(image.getPointer());
    resample_filter->Update();

    Image::Pointer resampled_image = resample_filter->GetOutput();
    resampled_image->DisconnectPipeline();
    resampled_image->SetOrigin(origin);
    resampled_image->SetSpacing(spacing);
    resampled_image->SetDirection(direction);

    return ITKImage(resampled_image);
}

ITKImage ResizeProcessor::process(ITKImage image, ITKImage::PixelType size_factor)
{
    if(image.depth > 1)
        return ResizeProcessor::process(image, size_factor, InterpolationMethod::Linear);

    typedef cv::Mat2d CVImage;

    CVImage cv_image(image.height, image.width);
    image.foreachPixel([&](uint x, uint y, uint, ITKImage::PixelType pixel) {
        cv_image.at<double>(y,x) = pixel;
    });

    int cv_interpolation_method = size_factor < 1 ? cv::INTER_AREA : cv::INTER_CUBIC;

    cv::Size cv_size;
    cv_size.width = std::ceil(image.width * size_factor);
    cv_size.height = std::ceil(image.height * size_factor);
    CVImage resized_cv_image(cv_size.height, cv_size.width);
    cv::resize(cv_image, resized_cv_image, resized_cv_image.size(),
               0, 0, cv_interpolation_method);

    ITKImage resized_image(resized_cv_image.cols, resized_cv_image.rows, 1);
    resized_image.setEachPixel([&](uint x, uint y, uint) {
        return resized_cv_image.at<double>(y,x);
    });
    return resized_image;
}

