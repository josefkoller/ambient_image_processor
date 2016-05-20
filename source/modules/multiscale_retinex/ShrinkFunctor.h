#ifndef SHRINKFUNCTOR_H
#define SHRINKFUNCTOR_H

#include <itkCovariantVector.h>

#include "ITKImage.h"

class ShrinkFunctor
{
public:
    ShrinkFunctor();
    typedef itk::CovariantVector<float, ITKImage::ImageDimension> PixelType;
    PixelType operator()(const PixelType& input); // TODO: make inline?
    void setLambda(float lambda);
private:
    float lambda;
};

#endif // SHRINKFUNCTOR_H
