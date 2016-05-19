#ifndef UNSHARPMASKINGPROCESSOR_H
#define UNSHARPMASKINGPROCESSOR_H

#include "ITKImage.h"

class UnsharpMaskingProcessor
{
private:
    UnsharpMaskingProcessor();
public:
    static ITKImage process(ITKImage source, uint kernel_size, float kernel_sigma, float factor);
};

#endif // UNSHARPMASKINGPROCESSOR_H
