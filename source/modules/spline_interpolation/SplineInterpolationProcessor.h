#ifndef SPLINEINTERPOLATIONPROCESSOR_H
#define SPLINEINTERPOLATIONPROCESSOR_H

#include "ITKImage.h"

class SplineInterpolationProcessor
{
private:
    SplineInterpolationProcessor();
public:

    struct ReferenceROIStatistic
    {
        ITKImage::PixelType median_value;
        int x;
        int y;
    };

    static ITKImage process(ITKImage image, uint spline_order,
                                 uint spline_levels, uint spline_control_points,
                                 std::vector<ReferenceROIStatistic> nodes,
                                 ITKImage& field_image);
    static void printMetric(std::vector<ReferenceROIStatistic> rois);
};

#endif // SPLINEINTERPOLATIONPROCESSOR_H
