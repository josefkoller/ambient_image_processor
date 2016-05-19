#include "HistogramProcessor.h"

#include <itkImageToHistogramFilter.h>

HistogramProcessor::HistogramProcessor()
{
}


void HistogramProcessor::calculate(const Image::Pointer& image,
                                   int bin_count,
                                   Image::PixelType window_from,
                                   Image::PixelType window_to,
                                   std::vector<double>& intensities,
                                   std::vector<double>& probabilities)
{
    typedef itk::Statistics::ImageToHistogramFilter<Image> HistogramGenerator;
    HistogramGenerator::Pointer histogram_generator = HistogramGenerator::New();

    HistogramGenerator::HistogramSizeType number_of_bins(1);
    number_of_bins[0] = bin_count;
    histogram_generator->SetHistogramSize(number_of_bins);

    histogram_generator->SetAutoMinimumMaximum(true);
    //   histogram_generator->SetHistogramBinMinimum(window_from);
    //   histogram_generator->SetHistogramBinMaximum(window_to);

    //  histogram_generator->SetClipBinsAtEnds(true);
    histogram_generator->SetMarginalScale(1);

    histogram_generator->SetInput(image);
    histogram_generator->Update();

    const HistogramGenerator::HistogramType *histogram = histogram_generator->GetOutput();

    /*
    const Image::SizeType size = image->GetLargestPossibleRegion().GetSize();
    long pixel_count = size[0] * size[1] * size[2];
    long samples_per_bin = ceil(((double)pixel_count) / bin_count); */
    Image::PixelType total_frequency = histogram->GetTotalFrequency();

    for(unsigned int i = 0; i < histogram->GetSize()[0]; i++)
    {
        double bin_min = histogram->GetBinMin(0, i);
        double bin_max = histogram->GetBinMax(0, i);

        if(bin_max < window_from || bin_min > window_to)
        {
            continue;
        }
        double intensity = bin_min + (bin_max - bin_min) * 0.5f;
        double probability = histogram->GetFrequency(i) / total_frequency;

        intensities.push_back(intensity);
        probabilities.push_back(probability);
    }
}

HistogramProcessor::Image::PixelType HistogramProcessor::minPixel(const Image::Pointer& image)
{
    if(image.IsNull())
        return 0;

    Image::PixelType minimum = 1e7;
    auto itk_image = ITKImage(image);
    itk_image.foreachPixel([&minimum](uint x, uint y, Image::PixelType value) {
        if(value < minimum)
            minimum = value;
    });
    return minimum;
}

HistogramProcessor::Image::PixelType HistogramProcessor::maxPixel(const Image::Pointer& image)
{
    if(image.IsNull())
        return 0;

    Image::PixelType maximum = -1e7;
    auto itk_image = ITKImage(image);
    itk_image.foreachPixel([&maximum](uint x, uint y, Image::PixelType value) {
        if(value > maximum)
            maximum = value;
    });
    return maximum;
}
