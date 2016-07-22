#include "ImageInformationProcessor.h"

#include <itkStatisticsImageFilter.h>

ImageInformationProcessor::ImageInformationProcessor()
{
}


ImageInformationProcessor::InformationMap ImageInformationProcessor::collectInformation(
        ITKImage itk_image)
{
    InformationMap information = InformationMap();

    if(itk_image.isNull())
        return information;

    typedef ITKImage::InnerITKImage Image;
    Image::Pointer image = itk_image.getPointer();

    Image::RegionType region = image->GetLargestPossibleRegion();
    Image::SizeType size = region.GetSize();
    Image::SpacingType spacing = image->GetSpacing();
    Image::PointType origin = image->GetOrigin();

    QString dimensions_info = "";
    QString origin_text = "";
    QString spacing_text = "";
    int voxel_count = 1;
    for(int dimension = 0; dimension < size.GetSizeDimension(); dimension++)
    {
        dimensions_info += QString::number(size[dimension]);
        origin_text += QString::number(origin[dimension]);
        spacing_text += QString::number(spacing[dimension]);
        voxel_count *= size[dimension];
        if(dimension < size.GetSizeDimension() - 1)
        {
            dimensions_info += " x ";
            spacing_text += " | ";
            origin_text += " | ";
        }
    }
    information.insert("dimensions", dimensions_info);
    information.insert("origin", origin_text);
    information.insert("spacing", spacing_text);
    information.insert("voxel_count", QString::number(voxel_count) );


    typedef itk::StatisticsImageFilter<Image> StatisticsCalculator;
    StatisticsCalculator::Pointer statistics_calculator = StatisticsCalculator::New();
    statistics_calculator->SetInput(image);
    int number_of_histogram_bins = ceil(sqrt(voxel_count));
    statistics_calculator->Update();

    information.insert("mean", QString::number(statistics_calculator->GetMean()));
    information.insert("standard_deviation", QString::number(statistics_calculator->GetSigma()));
    information.insert("variance", QString::number(statistics_calculator->GetVariance()));
    auto coefficient_of_variation =  statistics_calculator->GetSigma() / statistics_calculator->GetMean();
    information.insert("coefficient_of_variation", QString::number(coefficient_of_variation));
    information.insert("minimum", QString::number(statistics_calculator->GetMinimum()));
    information.insert("maximum", QString::number(statistics_calculator->GetMaximum()));

    ITKImage::PixelType sum_of_squared_values = 0;
    ITKImage::PixelType sum_of_absolute_values = 0;
    itk_image.foreachPixel([&](uint,uint,uint,ITKImage::PixelType value) {
        sum_of_absolute_values+= std::abs(value);
        sum_of_squared_values+= value*value;
    });
    information.insert("sum_of_absolute_values", QString::number(sum_of_absolute_values));
    information.insert("sum_of_squared_values", QString::number(sum_of_squared_values));


    return information;
}

double ImageInformationProcessor::coefficient_of_variation(ITKImage itk_image)
{
    typedef itk::StatisticsImageFilter<ITKImage::InnerITKImage> StatisticsCalculator;
    StatisticsCalculator::Pointer statistics_calculator = StatisticsCalculator::New();
    statistics_calculator->SetInput(itk_image.getPointer());
    statistics_calculator->Update();

    return statistics_calculator->GetSigma() / statistics_calculator->GetMean();
}
