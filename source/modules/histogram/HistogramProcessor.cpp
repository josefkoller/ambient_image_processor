#include "HistogramProcessor.h"

template<typename Pixel>
Pixel* kernel_density_estimation_kernel_launch(Pixel* image_pixels,
                                              uint voxel_count,
                                              uint spectrum_bandwidth,
                                              Pixel kernel_bandwidth,
                                              uint kernel_type,
                                              Pixel window_from,
                                              Pixel window_to);

HistogramProcessor::HistogramProcessor()
{
}


void HistogramProcessor::calculate(ITKImage image,
                                   uint spectrum_bandwidth,
                                   ITKImage::PixelType kernel_bandwidth,
                                   KernelType kernel_type,
                                   ITKImage::PixelType window_from,
                                   ITKImage::PixelType window_to,
                                   std::vector<double>& intensities,
                                   std::vector<double>& probabilities)
{
    if(image.voxel_count > 6e5)
    {
        std::cout << "voxel count too high for KDE algorithm (would take too long)" << std::endl;
        calculateFast(image,
                       spectrum_bandwidth,
                       window_from,
                       window_to,
                       intensities,
                       probabilities);
        return;
    }

    auto image_pixels = image.cloneToPixelArray();
    ITKImage::PixelType* spectrum = kernel_density_estimation_kernel_launch(image_pixels,
                                                             image.voxel_count,
                                                             spectrum_bandwidth,
                                                             kernel_bandwidth,
                                                             (uint) kernel_type,
                                                             window_from, window_to);
    delete[] image_pixels;

    ITKImage::PixelType sum = 0;
    for(uint a = 0; a < spectrum_bandwidth; a++)
        sum += spectrum[a];
    if(sum == 0)
        sum = 1;

    ITKImage::PixelType delta_spectrum = (window_to - window_from) / spectrum_bandwidth;
    intensities.clear();
    probabilities.clear();
    for(uint a = 0; a < spectrum_bandwidth; a++)
    {
        probabilities.push_back(spectrum[a] / sum);
        intensities.push_back(window_from + a * delta_spectrum);
    }

    delete[] spectrum;
}

double HistogramProcessor::calculateEntropy(const std::vector<double>& probabilities)
{
    ITKImage::PixelType entropy = 0;

    for(double probability : probabilities)
    {
        if(probability < 1e-5)
            continue;

        entropy += probability * std::log2(probability);
    }

    return -entropy;
}

void HistogramProcessor::calculateFast(ITKImage image,
               uint spectrum_bandwidth,
               ITKImage::PixelType window_from,
               ITKImage::PixelType window_to,
               std::vector<double>& intensities,
               std::vector<double>& probabilities)
{
    ITKImage::PixelType const spectrum_spawn = window_to - window_from;
    ITKImage::PixelType const delta_spectrum = spectrum_spawn / spectrum_bandwidth;
    for(uint a = 0; a < spectrum_bandwidth; a++)
    {
        intensities.push_back(window_from + delta_spectrum * a);
        probabilities.push_back(0);
    }

    image.foreachPixel([&probabilities, window_from, delta_spectrum]
                       (uint, uint, uint, ITKImage::PixelType pixel) {
        int spectrum_index = (pixel - window_from) / delta_spectrum;
        if(spectrum_index < 0 || spectrum_index >= probabilities.size())
            return;
        probabilities[spectrum_index] ++;
    });

    ITKImage::PixelType sum = 0;
    for(uint a = 0; a < spectrum_bandwidth; a++)
        sum += probabilities[a];

    for(uint a = 0; a < spectrum_bandwidth; a++)
        probabilities[a] /= sum;
}
