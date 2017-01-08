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

#include "HistogramProcessor.h"

template<typename Pixel>
Pixel* kernel_density_estimation_kernel_launch(Pixel* image_pixels,
                                               Pixel* mask_pixels,
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
                                   ITKImage mask,
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
        calculateFast(image, mask,
                       spectrum_bandwidth,
                       window_from,
                       window_to,
                       intensities,
                       probabilities);
        return;
    }

    auto image_pixels = image.cloneToPixelArray();
    auto mask_pixels = mask.cloneToPixelArray();
    ITKImage::PixelType* spectrum = kernel_density_estimation_kernel_launch(image_pixels,
                                                                            mask_pixels,
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


double HistogramProcessor::calculateEntropy(const ITKImage& image,const ITKImage& mask, ITKImage::PixelType kde_bandwidth)
{
    typedef ITKImage::PixelType Pixel;
    Pixel min, max;
    image.minimumAndMaximum(min, max);

    return calculateEntropy(image, mask, kde_bandwidth, min, max);
}

double HistogramProcessor::calculateEntropy(const ITKImage& image, const ITKImage& mask,
                                            ITKImage::PixelType kde_bandwidth,
                                            const ITKImage::PixelType min, const ITKImage::PixelType max)
{
    typedef ITKImage::PixelType Pixel;

    const uint spectrum_bandwidth = std::ceil(std::sqrt(image.voxel_count));
    if(kde_bandwidth < 0)
        kde_bandwidth = (max - min) / spectrum_bandwidth;

    Pixel* image_pixels = image.cloneToPixelArray();
    Pixel* mask_pixels = mask.cloneToPixelArray();
    Pixel* spectrum = kernel_density_estimation_kernel_launch(
       image_pixels, mask_pixels, image.voxel_count, spectrum_bandwidth, kde_bandwidth,
       3, // kernel type = Epanechnik
       min, max);
    delete[] image_pixels;

    Pixel sum = 0;
    for(uint a = 0; a < spectrum_bandwidth; a++)
        sum += spectrum[a];
    if(sum == 0)
        sum = 1;

    Pixel entropy = 0;
    for(uint a = 0; a < spectrum_bandwidth; a++)
    {
        Pixel probability = spectrum[a] / sum;
        if(probability < 1e-5)
            continue;
        entropy += probability * std::log2(probability);
    }

    delete[] spectrum;
    return -entropy;
}

void HistogramProcessor::calculateFast(ITKImage image,
                                       ITKImage mask,
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

    image.foreachPixel([&probabilities, window_from, delta_spectrum, &mask]
                       (uint x, uint y, uint z, ITKImage::PixelType pixel) {
        if(!mask.isNull() && mask.getPixel(x,y,z) == 0)
            return;

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
