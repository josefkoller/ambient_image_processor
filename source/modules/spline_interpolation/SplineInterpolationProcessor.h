#ifndef SPLINEINTERPOLATIONPROCESSOR_H
#define SPLINEINTERPOLATIONPROCESSOR_H

#include "ITKImage.h"

class SplineInterpolationProcessor
{
private:
    SplineInterpolationProcessor();

public:
    static ITKImage process(ITKImage source_image);
};

#endif // SPLINEINTERPOLATIONPROCESSOR_H
