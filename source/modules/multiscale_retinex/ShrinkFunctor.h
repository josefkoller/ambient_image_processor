#ifndef SHRINKFUNCTOR_H
#define SHRINKFUNCTOR_H

#include <itkCovariantVector.h>
#include <ITKImageProcessor.h>

class ShrinkFunctor
{
public:
    ShrinkFunctor();
    typedef itk::CovariantVector<float, InputDimension> PixelType;
    PixelType operator()(const PixelType& input); // TODO: make inline?
    void setLambda(float lambda);
private:
    float lambda;
};

#endif // SHRINKFUNCTOR_H
