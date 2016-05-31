#ifndef CUDAIMAGEOPERATIONSPROCESSOR_H
#define CUDAIMAGEOPERATIONSPROCESSOR_H

#include "ITKImage.h"
#include <functional>

class CudaImageOperationsProcessor
{
private:
    CudaImageOperationsProcessor();

    typedef ITKImage::PixelType Pixel;
    typedef Pixel* Pixels;
    typedef std::function<Pixels(Pixels, Pixels)> BinaryPixelsOperation;

    static ITKImage perform(ITKImage image1, ITKImage image2, BinaryPixelsOperation operation);
public:
    static ITKImage multiply(ITKImage image1, ITKImage image2);
    static ITKImage divide(ITKImage image1, ITKImage image2);
};

#endif // CUDAIMAGEOPERATIONSPROCESSOR_H
