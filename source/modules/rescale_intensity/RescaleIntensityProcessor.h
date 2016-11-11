#ifndef RESCALEINTENSITYPROCESSOR_H
#define RESCALEINTENSITYPROCESSOR_H

#include "ITKImage.h"

class RescaleIntensityProcessor
{
private:
    RescaleIntensityProcessor();

    static ITKImage process(ITKImage image, ITKImage::PixelType from, ITKImage::PixelType to,
                            ITKImage::PixelType minimum, ITKImage::PixelType maximum);
public:
    static ITKImage process(ITKImage image, ITKImage::PixelType from, ITKImage::PixelType to);
    static ITKImage process(ITKImage image, ITKImage::PixelType from, ITKImage::PixelType to,
                            ITKImage mask);
};

#endif // RESCALEINTENSITYPROCESSOR_H
