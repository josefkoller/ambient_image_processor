#ifndef SHRINKFILTER_H
#define SHRINKFILTER_H

#include <itkUnaryFunctorImageFilter.h>
#include "ShrinkFunctor.h"

typedef itk::Image< ShrinkFunctor::PixelType, InputDimension> VectorImageType;

class ShrinkFilter :
        public itk::UnaryFunctorImageFilter<VectorImageType, VectorImageType, ShrinkFunctor>
{
public:
    ShrinkFilter();

    typedef ShrinkFilter Self;
    typedef itk::UnaryFunctorImageFilter<VectorImageType, VectorImageType, ShrinkFunctor> Superclass;
    typedef itk::SmartPointer< Self > Pointer;
    itkNewMacro(Self);

    void setLambda(float lambda);
};

#endif // SHRINKFILTER_H
