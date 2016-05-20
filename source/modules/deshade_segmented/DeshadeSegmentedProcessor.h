#ifndef DESHADESEGMENTEDPROCESSOR_H
#define DESHADESEGMENTEDPROCESSOR_H

#include <itkImage.h>
#include <itkLabelStatisticsImageFilter.h>

#include "ITKImage.h"

#include "RegionGrowingSegmentationProcessor.h"

class DeshadeSegmentedProcessor
{
public:
    typedef RegionGrowingSegmentationProcessor::LabelImage LabelImage;
    typedef itk::Image<unsigned char, ITKImage::ImageDimension> ITKLabelImage;

private:
    DeshadeSegmentedProcessor();

    typedef ITKImage::InnerITKImage Image;
    typedef Image::IndexType SeedPoint;
    typedef std::vector<SeedPoint> Segment;
    typedef std::vector<Segment> Segments;

    typedef itk::LabelStatisticsImageFilter<Image, ITKLabelImage> StatisticsFilter;

    static void computeReflectanceInSegment(Segment segment,
        LabelImage label_image, ITKImage reflectance_image,
        StatisticsFilter::Pointer statistics_filter);
public:

    static ITKImage process(
      ITKImage source_image, float lambda, Segments segments,
      LabelImage label_image, ITKImage& reflectance_image);
};

#endif // DESHADESEGMENTEDPROCESSOR_H
