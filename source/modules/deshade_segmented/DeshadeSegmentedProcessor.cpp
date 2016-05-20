#include "DeshadeSegmentedProcessor.h"

#include <itkImageRegionIteratorWithIndex.h>
#include <itkSubtractImageFilter.h>
#include <itkCastImageFilter.h>

DeshadeSegmentedProcessor::DeshadeSegmentedProcessor()
{

}

void DeshadeSegmentedProcessor::computeReflectanceInSegment(Segment segment,
    LabelImage label_image, ITKImage reflectance_image,
    StatisticsFilter::Pointer statistics_filter)
{
    SeedPoint first_seed_point = segment[0];
    uint segment_label = label_image.getPixel(first_seed_point);

    float reflectance = statistics_filter->GetMean(segment_label);

    label_image.foreachPixel([&reflectance_image, segment_label, reflectance](
                             uint x,uint y, uint z, ITKImage::PixelType label_pixel) {
        if(label_pixel == segment_label)
            reflectance_image.setPixel(x, y, z, reflectance);
    });
}

ITKImage DeshadeSegmentedProcessor::process(
  ITKImage source_image, float lambda, Segments segments,
  LabelImage label_image, ITKImage& reflectance_image)
{
    // reflectance is the mean inside the segments
    reflectance_image = source_image.cloneSameSizeWithZeros();

    typedef itk::CastImageFilter<LabelImage::InnerITKImage, ITKLabelImage> CastLabelFilter;
    CastLabelFilter::Pointer cast_label_filter = CastLabelFilter::New();
    cast_label_filter->SetInput(label_image.getPointer());
    cast_label_filter->Update();
    ITKLabelImage::Pointer itk_label_image = cast_label_filter->GetOutput();
    itk_label_image->DisconnectPipeline();

    StatisticsFilter::Pointer statistics_filter = StatisticsFilter::New();
    statistics_filter->SetInput(source_image.getPointer());
    statistics_filter->SetLabelInput(itk_label_image);
    statistics_filter->Update();

    for(Segment segment : segments)
    {
        computeReflectanceInSegment(segment, label_image, reflectance_image, statistics_filter);
    }

    typedef itk::SubtractImageFilter<Image> SubtractFilter;
    SubtractFilter::Pointer subtract_filter = SubtractFilter::New();
    subtract_filter->SetInput1(source_image.getPointer());
    subtract_filter->SetInput2(reflectance_image.getPointer());
    subtract_filter->Update();
    Image::Pointer shading_image = subtract_filter->GetOutput();
    shading_image->DisconnectPipeline();
    return ITKImage(shading_image);
}
