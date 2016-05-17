#ifndef NONLOCALGRADIENTPROCESSOR_H
#define NONLOCALGRADIENTPROCESSOR_H

#include <itkImage.h>


class NonLocalGradientProcessor
{
private:
    NonLocalGradientProcessor();

public:
    typedef itk::Image<double> Image;

    static Image::Pointer process(Image::Pointer source,
                                  uint kernel_size,
                                  Image::PixelType kernel_sigma);
private:
    static Image::PixelType* createKernel(
            uint kernel_size,
            Image::PixelType kernel_sigma);
};

#endif // NONLOCALGRADIENTPROCESSOR_H
