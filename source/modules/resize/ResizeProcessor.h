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
        Sinc,
        BSpline3
    };

    static ITKImage process(ITKImage image,
                            ITKImage::PixelType size_factor, InterpolationMethod interpolation_method);

    static ITKImage process(ITKImage image,
                            uint width, uint height, uint depth,
                            InterpolationMethod interpolationMethod);

    static ITKImage process(ITKImage image,
                            uint width, uint height, uint depth);
};

#endif // RESIZEPROCESSOR_H
