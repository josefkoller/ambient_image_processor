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

#include "NormalizeStatisticProcessor.h"

#include "CudaImageOperationsProcessor.h"

#include <itkStatisticsImageFilter.h>

typedef ITKImage::InnerITKImage Image;
typedef itk::StatisticsImageFilter<Image> StatisticsCalculator;

NormalizeStatisticProcessor::NormalizeStatisticProcessor()
{
}

ITKImage NormalizeStatisticProcessor::equalizeMeanAddConstant(ITKImage image, ITKImage reference_image)
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

ITKImage NormalizeStatisticProcessor::equalizeMeanScale(ITKImage image, ITKImage reference_image)
{
    StatisticsCalculator::Pointer statistics_calculator = StatisticsCalculator::New();
    statistics_calculator->SetInput(image.getPointer());
    statistics_calculator->Update();

    ITKImage::PixelType image_mean = statistics_calculator->GetMean();
    statistics_calculator->SetInput(reference_image.getPointer());
    statistics_calculator->Update();
    ITKImage::PixelType reference_image_mean = statistics_calculator->GetMean();

    auto ratio = reference_image_mean / image_mean;

    return CudaImageOperationsProcessor::multiplyConstant(image, ratio);
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

