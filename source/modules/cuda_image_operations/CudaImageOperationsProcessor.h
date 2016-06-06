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
    typedef std::function<Pixels(Pixels)> UnaryPixelsOperation;

    static ITKImage perform(ITKImage image1, ITKImage image2, BinaryPixelsOperation operation);
    static ITKImage perform(ITKImage image, UnaryPixelsOperation operation);
public:
    static ITKImage multiply(ITKImage image1, ITKImage image2);
    static ITKImage divide(ITKImage image1, ITKImage image2);
    static ITKImage add(ITKImage image1, ITKImage image2);
    static ITKImage subtract(ITKImage image1, ITKImage image2);

    static ITKImage addConstant(ITKImage image, ITKImage::PixelType constant);

    static ITKImage convolution3x3(ITKImage image, ITKImage::PixelType* kernel);
    static ITKImage convolution3x3x3(ITKImage image, ITKImage::PixelType* kernel, bool correct_center);
};

#endif // CUDAIMAGEOPERATIONSPROCESSOR_H
