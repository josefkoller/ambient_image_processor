#ifndef DESHADESEGMENTEDPROCESSOR_H
#define DESHADESEGMENTEDPROCESSOR_H

#include <itkImage.h>
#include <itkLabelStatisticsImageFilter.h>

#include "ITKImage.h"

class DeshadeSegmentedProcessor
{
private:
    DeshadeSegmentedProcessor();

    typedef ITKImage::InnerITKImage Image;
    typedef Image::IndexType SeedPoint;
    typedef std::vector<SeedPoint> Segment;
    typedef std::vector<Segment> Segments;
    typedef itk::Image<unsigned char, Image::ImageDimension> LabelImage;

    typedef itk::LabelStatisticsImageFilter<Image, LabelImage> StatisticsFilter;

    static void computeReflectanceInSegment(Segment segment,
        LabelImage::Pointer label_image, Image::Pointer reflectance_image,
        StatisticsFilter::Pointer statistics_filter);

    static void initializeReflectanceImage(Image::Pointer reflectance_image);
public:
    static Image::Pointer process(
      Image::Pointer source_image, float lambda, Segments segments,
      LabelImage::Pointer label_image, Image::Pointer& reflectance_image);
};

#endif // DESHADESEGMENTEDPROCESSOR_H
