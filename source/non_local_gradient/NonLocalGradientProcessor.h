#ifndef NONLOCALGRADIENTPROCESSOR_H
#define NONLOCALGRADIENTPROCESSOR_H

#include <itkImage.h>


class NonLocalGradientProcessor
{
private:
    NonLocalGradientProcessor();

public:
    typedef itk::Image<float> Image;

    static Image::Pointer process(Image::Pointer source,
                                  uint kernel_size,
                                  float kernel_sigma);
private:
    static float* createKernel(
            uint kernel_size,
            float kernel_sigma);
};

#endif // NONLOCALGRADIENTPROCESSOR_H
