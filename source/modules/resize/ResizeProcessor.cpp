/*
    Ambient Image Processor - A tool to perform several imaging tasks
    
    Copyright (C) 2016 Josef Koller

    https://github.com/josefkoller/ambient_image_processor    
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "ResizeProcessor.h"

#include "ExtractProcessor.h"

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
    uint depth = std::ceil(image.depth * size_factor);

    width = std::max(1u, width);
    height = std::max(1u, height);
    depth = std::max(1u, depth);

    return process(image, width, height, depth, interpolation_method);
}

ITKImage ResizeProcessor::process(ITKImage image,
                                  uint width, uint height, uint depth,
                                  ResizeProcessor::InterpolationMethod interpolation_method)
{
    // FIX itk bug: if upsampling make the image one pixel bigger and extract afterwards
    // otherwise the last pixel row in reach dimension would contain some bright pixels
    const uint additional_size = 2;

    const bool is_upsampling = width > image.width || height > image.height || depth > image.depth;
    if(is_upsampling) {
        width+= additional_size;
        height+= additional_size;
        depth+= additional_size;
        std::cout << "upsampling" << std::endl;
    }

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

    ITKImage::PixelType size_factor_x = width / ((ITKImage::PixelType)image.width);
    ITKImage::PixelType size_factor_y = height / ((ITKImage::PixelType)image.height);
    ITKImage::PixelType size_factor_z = depth / ((ITKImage::PixelType)image.depth);

    Image::SpacingType image_spacing = image.getPointer()->GetSpacing();
    Image::SpacingType spacing;
    spacing[0] = image_spacing[0] / size_factor_x;  // physical size keeps the same !!
    spacing[1] = image_spacing[1] / size_factor_y;
    spacing[2] = image_spacing[2] / size_factor_z;

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

    try {
        resample_filter->Update();

    } catch(itk::ExceptionObject exception) {
        std::cerr << "ResizeProcessor failed: " << exception.GetDescription() << std::endl;
        throw exception;
    }

    Image::Pointer resampled_image = resample_filter->GetOutput();
    resampled_image->DisconnectPipeline();

    image.getPointer()->SetOrigin(original_origin);
    resampled_image->SetOrigin(original_origin);

    auto resized_image = ITKImage(resampled_image);

    if(is_upsampling) {
        resized_image = ExtractProcessor::process(resized_image,
                                                  0, width - 1 - additional_size,
                                                  0, height - 1 - additional_size,
                                                  0, depth - 1 - additional_size);
    }

    return resized_image;
}

ITKImage ResizeProcessor::process(ITKImage image,
                                  uint width, uint height, uint depth)
{
    if(image.isNull())
        return ITKImage();

    // 3D ...
    if(image.depth > 1)
        return ResizeProcessor::process(image, width, height, depth, InterpolationMethod::Linear);

    // 2D ...
    typedef cv::Mat2d CVImage;

    CVImage cv_image(image.height, image.width);
    image.foreachPixel([&](uint x, uint y, uint, ITKImage::PixelType pixel) {
        cv_image.at<double>(y,x) = pixel;
    });

    ITKImage::PixelType size_factor = image.width / ((ITKImage::PixelType)width);

    int cv_interpolation_method = size_factor < 1 ? cv::INTER_AREA : cv::INTER_CUBIC;

    cv::Size cv_size;
    cv_size.width = width;
    cv_size.height = height;
    CVImage resized_cv_image(cv_size.height, cv_size.width);
    cv::resize(cv_image, resized_cv_image, resized_cv_image.size(),
               0, 0, cv_interpolation_method);

    ITKImage resized_image(resized_cv_image.cols, resized_cv_image.rows, 1);
    resized_image.setEachPixel([&](uint x, uint y, uint) {
        return resized_cv_image.at<double>(y,x);
    });
    return resized_image;
}

