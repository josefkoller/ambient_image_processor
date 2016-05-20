#ifndef NONLOCALGRADIENTPROCESSOR_H
#define NONLOCALGRADIENTPROCESSOR_H

#include <ITKImage.h>

class NonLocalGradientProcessor
{
private:
    NonLocalGradientProcessor();

public:

    static ITKImage process(ITKImage source,
                            uint kernel_size,
                            ITKImage::PixelType kernel_sigma);
private:
    static ITKImage createKernel(
            uint kernel_size,
            ITKImage::PixelType kernel_sigma);
};

#endif // NONLOCALGRADIENTPROCESSOR_H
