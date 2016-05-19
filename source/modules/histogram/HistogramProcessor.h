#ifndef HISTOGRAMPROCESSOR_H
#define HISTOGRAMPROCESSOR_H

#include "ITKImage.h"
#include <vector>

class HistogramProcessor
{
private:
    HistogramProcessor();

    typedef ITKImage::InnerITKImage Image;
public:
    static void calculate(const Image::Pointer& image,
                          int bin_count,
                          Image::PixelType window_from,
                          Image::PixelType window_to,
                          std::vector<double>& intensities,
                          std::vector<double>& probabilities);

    static Image::PixelType minPixel(const Image::Pointer& image);
    static Image::PixelType maxPixel(const Image::Pointer& image);
};

#endif // HISTOGRAMPROCESSOR_H
