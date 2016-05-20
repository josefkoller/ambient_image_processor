#ifndef HISTOGRAMPROCESSOR_H
#define HISTOGRAMPROCESSOR_H

#include "ITKImage.h"
#include <vector>

class HistogramProcessor
{
private:
    HistogramProcessor();

public:
    static void calculate(ITKImage image,
                          int bin_count,
                          ITKImage::PixelType window_from,
                          ITKImage::PixelType window_to,
                          std::vector<double>& intensities,
                          std::vector<double>& probabilities);
};

#endif // HISTOGRAMPROCESSOR_H
