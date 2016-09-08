#include "ResizeProcessor.h"

#include <itkResampleImageFilter.h>

#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkWindowedSincInterpolateImageFunction.h>
#include <itkBSplineInterpolateImageFunction.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

ResizeProcessor::ResizeProcessor()
{
}

ITKImage ResizeProcessor::process(ITKImage image,
                                  ITKImage::PixelType size_factor,
                                  ResizeProcessor::InterpolationMethod interpolation_method)
{
    uint width = std::ceil(image.width * size_factor);
    uint height = std::ceil(image.height * size_factor);
    uint depth = image.depth == 1 ? 1 : std::ceil(image.depth * size_factor);

    return process(image, size_factor, width, height, depth, interpolation_method);
}

ITKImage ResizeProcessor::process(ITKImage image, ITKImage::PixelType size_factor,
                                  uint width, uint height, uint depth,
                                  ResizeProcessor::InterpolationMethod interpolation_method)
{
    typedef ITKImage::InnerITKImage Image;
    typedef itk::ResampleImageFilter<Image, Image> ResampleFilter;

    typedef itk::NearestNeighborInterpolateImageFunction<Image> NearestNeighborInterpolation;
    typedef itk::LinearInterpolateImageFunction<Image> LinearInterpolation;
    typedef itk::WindowedSincInterpolateImageFunction<Image, 4> SincInterpolation;
    typedef itk::BSplineInterpolateImageFunction<Image> BSplineInterpolation;

    Image::PointType original_origin = image.getPointer()->GetOrigin();
    Image::PointType origin;
    origin.Fill(0);
    image.getPointer()->SetOrigin(origin); // setting input origin to zero for processing

    Image::SpacingType image_spacing = image.getPointer()->GetSpacing();
    Image::SpacingType spacing;
    spacing[0] = image_spacing[0] / size_factor;  // physical size keeps the same !!
    spacing[1] = image_spacing[1] / size_factor;
    spacing[2] = image_spacing[2] / size_factor;

    std::cout << "spacing from : " << image_spacing << std::endl;
    std::cout << "to : " << spacing << std::endl;

    Image::DirectionType direction;
    direction.SetIdentity();

    Image::SizeType image_size = image.getPointer()->GetLargestPossibleRegion().GetSize();
    Image::SizeType size;
    size[0] = width;
    size[1] = height;
    size[2] = depth;

    std::cout << "size from : " << image_size << std::endl;
    std::cout << "to : " << size << std::endl;

    ResampleFilter::Pointer resample_filter = ResampleFilter::New();

    if(interpolation_method == InterpolationMethod::NearestNeighbour)
        resample_filter->SetInterpolator(NearestNeighborInterpolation::New());
    else if(interpolation_method == InterpolationMethod::Linear)
        resample_filter->SetInterpolator(LinearInterpolation::New());
    else if(interpolation_method == InterpolationMethod::Sinc)
        resample_filter->SetInterpolator(SincInterpolation::New());
    else
    {
        BSplineInterpolation::Pointer interpolation = BSplineInterpolation::New();
        interpolation->SetSplineOrder(3);
        resample_filter->SetInterpolator(interpolation);
    }

    resample_filter->SetOutputOrigin(origin);
    resample_filter->SetOutputSpacing(spacing);
    resample_filter->SetOutputDirection(direction);
    resample_filter->SetSize(size);
    resample_filter->SetInput(image.getPointer());
    resample_filter->Update();

    Image::Pointer resampled_image = resample_filter->GetOutput();
    resampled_image->DisconnectPipeline();

    image.getPointer()->SetOrigin(original_origin);
    resampled_image->SetOrigin(original_origin);

    return ITKImage(resampled_image);
}

ITKImage ResizeProcessor::process(ITKImage image, ITKImage::PixelType size_factor)
{
    if(image.isNull())
        return ITKImage();

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

