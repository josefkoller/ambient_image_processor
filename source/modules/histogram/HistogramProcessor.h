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

#ifndef HISTOGRAMPROCESSOR_H
#define HISTOGRAMPROCESSOR_H

#include "ITKImage.h"
#include <vector>

class HistogramProcessor
{
private:
    HistogramProcessor();

public:

    enum KernelType {
        Uniform = 0,
        Gaussian,
        Cosine,
        Epanechnik
    };

    static void calculate(ITKImage image,
                          ITKImage mask,
                          uint spectrum_bandwidth,
                          ITKImage::PixelType kernel_bandwidth,
                          KernelType kernel_type,
                          ITKImage::PixelType window_from,
                          ITKImage::PixelType window_to,
                          std::vector<double>& intensities,
                          std::vector<double>& probabilities);

    static void calculateFast(ITKImage image,
                              ITKImage mask,
                          uint spectrum_bandwidth,
                          ITKImage::PixelType window_from,
                          ITKImage::PixelType window_to,
                          std::vector<double>& intensities,
                          std::vector<double>& probabilities);

    static double calculateEntropy(const std::vector<double>& probabilities);
    static double calculateEntropy(const ITKImage& image, const ITKImage& mask, const ITKImage::PixelType kde_bandwidth);
    static double calculateEntropy(const ITKImage& image, const ITKImage& mask, const ITKImage::PixelType kde_bandwidth,
                                   const ITKImage::PixelType window_from, const ITKImage::PixelType window_to);
};

#endif // HISTOGRAMPROCESSOR_H
