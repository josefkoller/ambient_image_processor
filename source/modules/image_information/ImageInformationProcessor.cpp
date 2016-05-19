#include "ImageInformationProcessor.h"

#include <itkStatisticsImageFilter.h>

ImageInformationProcessor::ImageInformationProcessor()
{
}


ImageInformationProcessor::InformationMap ImageInformationProcessor::collectInformation(
        ITKImage::InnerITKImage::Pointer image)
{
    InformationMap information = InformationMap();

    if(image.IsNull())
        return information;

    typedef ITKImage::InnerITKImage Image;

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
    auto standard_error =  statistics_calculator->GetVariance() / statistics_calculator->GetMean();
    information.insert("standard_error", QString::number(standard_error));
    information.insert("minimum", QString::number(statistics_calculator->GetMinimum()));
    information.insert("maximum", QString::number(statistics_calculator->GetMaximum()));

/*
    this->ui->window_from_spinbox->setMinimum(statistics_calculator->GetMinimum());
    this->ui->window_from_spinbox->setMaximum(statistics_calculator->GetMaximum());
    this->ui->window_from_spinbox->setValue(statistics_calculator->GetMinimum());

    this->ui->window_to_spinbox->setMinimum(statistics_calculator->GetMinimum());
    this->ui->window_to_spinbox->setMaximum(statistics_calculator->GetMaximum());
    this->ui->window_to_spinbox->setValue(statistics_calculator->GetMaximum());

    this->ui->fromMinimumButton->setText("From Minimum: " + QString::number(
                                             statistics_calculator->GetMinimum() ));
    this->ui->toMaximumButton->setText("To Maximum: " + QString::number(
                                             statistics_calculator->GetMaximum() ));
                                             */

    return information;
}
