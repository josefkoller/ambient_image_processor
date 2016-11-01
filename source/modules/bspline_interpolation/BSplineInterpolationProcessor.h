#ifndef BSPLINEINTERPOLATIONPROCESSOR_H
#define BSPLINEINTERPOLATIONPROCESSOR_H

#include "ITKImage.h"

class BSplineInterpolationProcessor
{
public:
    BSplineInterpolationProcessor();

    static ITKImage process(ITKImage image, ITKImage mask,
      uint spline_order, uint number_of_fitting_levels);

private:
    template<unsigned int NDimension = 3>
    static ITKImage processDimensions(ITKImage image, ITKImage mask,
                               uint spline_order, uint number_of_fitting_levels);
};

#endif // BSPLINEINTERPOLATIONPROCESSOR_H
