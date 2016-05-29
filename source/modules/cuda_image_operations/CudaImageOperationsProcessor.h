#ifndef CUDAIMAGEOPERATIONSPROCESSOR_H
#define CUDAIMAGEOPERATIONSPROCESSOR_H

#include "ITKImage.h"

class CudaImageOperationsProcessor
{
private:
    CudaImageOperationsProcessor();

public:
    static ITKImage multiply(ITKImage image1, ITKImage image2);
};

#endif // CUDAIMAGEOPERATIONSPROCESSOR_H
