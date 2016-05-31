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
    reflectance_image.setEachPixel([](uint,uint,uint) {
        return -1;
    });

    try
    {
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

        typedef itk::SubtractImageFilter<Image> ShadingExtractor;
        ShadingExtractor::Pointer shading_extractor = ShadingExtractor::New();
        shading_extractor->SetInput1(source_image.getPointer());
        shading_extractor->SetInput2(reflectance_image.getPointer());
        shading_extractor->Update();
        Image::Pointer shading_image = shading_extractor->GetOutput();
        shading_image->DisconnectPipeline();
        auto itk_shading_image = ITKImage(shading_image);

        itk_shading_image.setEachPixel([&itk_shading_image, &reflectance_image] (uint x, uint y, uint z) {
            if(reflectance_image.getPixel(x,y,z) == -1)
                return 0.0;
            else
                return itk_shading_image.getPixel(x,y,z);
        });
        return itk_shading_image;
    }
    catch(itk::ExceptionObject exception)
    {
        exception.Print(std::cerr);
        throw exception;
    }
    return ITKImage();
}
