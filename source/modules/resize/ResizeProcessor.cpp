#include "ResizeProcessor.h"

#include <itkResampleImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkScaleTransform.h>

#include <itkWindowedSincInterpolateImageFunction.h>

ResizeProcessor::ResizeProcessor()
{
}

ITKImage ResizeProcessor::process(ITKImage image,
                                  ITKImage::PixelType size_factor,
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
    for(int i = 0; i < Image::ImageDimension; i++)
        size[i] = std::max(1.0, image_size[i] * size_factor);

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

