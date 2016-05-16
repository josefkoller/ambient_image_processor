#include "DeshadeSegmentedProcessor.h"

#include <itkImageRegionIteratorWithIndex.h>

#include <itkSubtractImageFilter.h>

DeshadeSegmentedProcessor::DeshadeSegmentedProcessor()
{

}

void DeshadeSegmentedProcessor::computeReflectanceInSegment(Segment segment,
    LabelImage::Pointer label_image, Image::Pointer reflectance_image,
    StatisticsFilter::Pointer statistics_filter)
{
    SeedPoint first_seed_point = segment[0];
    uint segment_label = label_image->GetPixel(first_seed_point);

    float reflectance = statistics_filter->GetMean(segment_label);

    itk::ImageRegionIteratorWithIndex<Image> reflectance_iterator(reflectance_image,
                                                         reflectance_image->GetLargestPossibleRegion());
    while(!reflectance_iterator.IsAtEnd())
    {
        Image::IndexType index = reflectance_iterator.GetIndex();
        LabelImage::IndexType label_index = index;

        if(label_image->GetPixel(label_index) == segment_label)
        {
            reflectance_iterator.Set(reflectance);
        }

        ++reflectance_iterator;
    }
}

void DeshadeSegmentedProcessor::initializeReflectanceImage(Image::Pointer reflectance_image)
{
    itk::ImageRegionIteratorWithIndex<Image> reflectance_iterator(reflectance_image,
                                                         reflectance_image->GetLargestPossibleRegion());
    while(!reflectance_iterator.IsAtEnd())
    {
        reflectance_iterator.Set(0);

        ++reflectance_iterator;
    }
}

DeshadeSegmentedProcessor::Image::Pointer DeshadeSegmentedProcessor::process(
  Image::Pointer source_image, float lambda, Segments segments,
  LabelImage::Pointer label_image, Image::Pointer& reflectance_image)
{
    // reflectance is the mean inside the segments
    reflectance_image = Image::New();
    reflectance_image->SetRegions(source_image->GetLargestPossibleRegion());
    reflectance_image->Allocate();
    initializeReflectanceImage(reflectance_image);

    StatisticsFilter::Pointer statistics_filter = StatisticsFilter::New();
    statistics_filter->SetInput(source_image);
    statistics_filter->SetLabelInput(label_image);
    statistics_filter->Update();

    for(Segment segment : segments)
    {
        computeReflectanceInSegment(segment, label_image, reflectance_image, statistics_filter);
    }

    typedef itk::SubtractImageFilter<Image> SubtractFilter;
    SubtractFilter::Pointer subtract_filter = SubtractFilter::New();
    subtract_filter->SetInput1(source_image);
    subtract_filter->SetInput2(reflectance_image);

    subtract_filter->Update();
    Image::Pointer shading_image = subtract_filter->GetOutput();
    shading_image->DisconnectPipeline();


    return shading_image;
}
