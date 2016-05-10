#include "Optimizer.h"

#include "../CircleFactory.h"
#include "../CircleImage.h"

#include <itkMultiplyImageFilter.h>
#include <itkGradientMagnitudeImageFilter.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageToHistogramFilter.h>

Optimizer::Optimizer()
{

}

ITKImage::Pointer Optimizer::multiply(ITKImage::Pointer image1,
                                      ITKImage::Pointer image2)
{
    typedef itk::MultiplyImageFilter<ITKImage> Multiplier;
    Multiplier::Pointer multiplier = Multiplier::New();
    multiplier->SetInput1(image1);
    multiplier->SetInput2(image2);
    multiplier->Update();
    ITKImage::Pointer product = multiplier->GetOutput();
    product->DisconnectPipeline();
    return product;
}

ITKImage::Pointer Optimizer::applyRandom(ITKImage::Pointer input_image,
                                         ITKImage::Pointer& field_image)
{
    auto size = input_image->GetLargestPossibleRegion().GetSize();
    auto width = size[0];
    auto height = size[1];

    auto circle = CircleFactory::createRandom();
    auto circle_image = CircleImage(circle, width, height);
    field_image = ITKCircleImage::create(circle_image);

    return multiply(input_image, field_image);
}

float Optimizer::metric(ITKImage::Pointer image)
{
    return entropy(image);
}

float Optimizer::entropy(ITKImage::Pointer image)
{
    ITKImage::SizeType size = image->GetLargestPossibleRegion().GetSize();
    uint bin_count = size[0] * size[1];

    typedef itk::Statistics::ImageToHistogramFilter<ITKImage> HistogramFilter;
    HistogramFilter::Pointer histogram_filter = HistogramFilter::New();
    histogram_filter->SetInput(image);

    HistogramFilter::HistogramSizeType histogram_size(1);
    histogram_size[0] = bin_count;
    histogram_filter->SetHistogramSize(histogram_size);

    histogram_filter->Update();

    HistogramFilter::HistogramType* histogram = histogram_filter->GetOutput();
    const uint frequency_count = histogram->Size();
    float entropy = 0;
    for(uint i = 0; i < frequency_count; i++)
    {
        float probability = histogram->GetFrequency(i) /
                static_cast<float>(histogram->GetTotalFrequency());
        if(probability > 1e-7f)
        {
            entropy += probability * std::log2f(probability);
        }
    }
    entropy *= -1;
    std::cout << "entropy: " << entropy << std::endl;
    return entropy;
}

float Optimizer::tv(ITKImage::Pointer image)
{
    float tv = 0;
    typedef itk::GradientMagnitudeImageFilter<ITKImage, ITKImage> GradientFilter;
    GradientFilter::Pointer gradient_filter = GradientFilter::New();
    gradient_filter->SetInput(image);
    gradient_filter->Update();
    ITKImage::Pointer gradient = gradient_filter->GetOutput();
    gradient->DisconnectPipeline();

    itk::ImageRegionConstIterator<ITKImage> iterator(gradient, gradient->GetLargestPossibleRegion());
    while(!iterator.IsAtEnd())
    {
        tv += std::abs(iterator.Get());
        ++iterator;
    }

    return tv;
}

ITKImage::Pointer Optimizer::run(ITKImage::Pointer input_image, uint iteration_count,
                                 ITKImage::Pointer& best_field_image)
{
    ITKImage::Pointer best_image = this->applyRandom(input_image, best_field_image);
    float best_metric = this->metric(best_image);

    for(uint iteration = 0; iteration < iteration_count; iteration++)
    {
        ITKImage::Pointer field_image = NULL;
        auto image = this->applyRandom(input_image, field_image);
        auto image_metric = this->metric(image);

        if(image_metric < best_metric)
        {
            best_metric = image_metric;
            best_image = image;
            best_field_image = field_image;
        }
    }

    std::cout << "best entropy: " << best_metric << std::endl;

    return best_image;
}
