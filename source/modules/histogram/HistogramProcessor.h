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
                          uint spectrum_bandwidth,
                          ITKImage::PixelType kernel_bandwidth,
                          KernelType kernel_type,
                          ITKImage::PixelType window_from,
                          ITKImage::PixelType window_to,
                          std::vector<double>& intensities,
                          std::vector<double>& probabilities);

    static double calculateEntropy(const std::vector<double>& probabilities);
};

#endif // HISTOGRAMPROCESSOR_H
