#ifndef RESIZEPROCESSOR_H
#define RESIZEPROCESSOR_H

#include "ITKImage.h"

class ResizeProcessor
{
private:
    ResizeProcessor();

public:
    enum InterpolationMethod
    {
        NearestNeighbour = 0,
        Linear,
        Sinc
    };

    static ITKImage process(ITKImage image,
                            ITKImage::PixelType size_factor, InterpolationMethod interpolation_method);
};

#endif // RESIZEPROCESSOR_H
