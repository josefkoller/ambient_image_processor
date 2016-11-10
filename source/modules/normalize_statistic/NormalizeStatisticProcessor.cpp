#include "NormalizeStatisticProcessor.h"

#include "CudaImageOperationsProcessor.h"

#include <itkStatisticsImageFilter.h>

typedef ITKImage::InnerITKImage Image;
typedef itk::StatisticsImageFilter<Image> StatisticsCalculator;

NormalizeStatisticProcessor::NormalizeStatisticProcessor()
{
}

ITKImage NormalizeStatisticProcessor::equalizeMean(ITKImage image, ITKImage reference_image)
{
    StatisticsCalculator::Pointer statistics_calculator = StatisticsCalculator::New();
    statistics_calculator->SetInput(image.getPointer());
    statistics_calculator->Update();

    ITKImage::PixelType image_mean = statistics_calculator->GetMean();
    statistics_calculator->SetInput(reference_image.getPointer());
    statistics_calculator->Update();
    ITKImage::PixelType reference_image_mean = statistics_calculator->GetMean();

    auto difference = reference_image_mean - image_mean;

    return CudaImageOperationsProcessor::addConstant(image, difference);
}

ITKImage NormalizeStatisticProcessor::equalizeStandardDeviation(ITKImage image, ITKImage reference_image)
{
    StatisticsCalculator::Pointer statistics_calculator = StatisticsCalculator::New();
    statistics_calculator->SetInput(image.getPointer());
    statistics_calculator->Update();

    ITKImage::PixelType image_std = statistics_calculator->GetSigma();
    statistics_calculator->SetInput(reference_image.getPointer());
    statistics_calculator->Update();
    ITKImage::PixelType reference_image_std = statistics_calculator->GetSigma();

    auto ratio = reference_image_std / image_std;

    return CudaImageOperationsProcessor::multiplyConstant(image, ratio);
}

ITKImage NormalizeStatisticProcessor::equalizeMaxMin(ITKImage image, ITKImage reference_image)
{
    StatisticsCalculator::Pointer statistics_calculator = StatisticsCalculator::New();
    statistics_calculator->SetInput(image.getPointer());
    statistics_calculator->Update();

    ITKImage::PixelType image_bandwidth =
            statistics_calculator->GetMaximum() - statistics_calculator->GetMinimum();
    statistics_calculator->SetInput(reference_image.getPointer());
    statistics_calculator->Update();
    ITKImage::PixelType reference_image_bandwidth =
            statistics_calculator->GetMaximum() - statistics_calculator->GetMinimum();

    auto factor = reference_image_bandwidth / image_bandwidth;

    return CudaImageOperationsProcessor::multiplyConstant(image, factor);
}

