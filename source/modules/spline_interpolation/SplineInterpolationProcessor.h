#ifndef SPLINEINTERPOLATIONPROCESSOR_H
#define SPLINEINTERPOLATIONPROCESSOR_H

#include "ITKImage.h"

class SplineInterpolationProcessor
{
private:
    SplineInterpolationProcessor();

    typedef ITKImage::InnerITKImage ImageType;
public:

    struct ReferenceROIStatistic
    {
        ImageType::PixelType median_value;
        int x;
        int y;
    };

    static ImageType::Pointer process(ImageType::Pointer image, uint spline_order,
                                 uint spline_levels, uint spline_control_points,
                                 std::vector<ReferenceROIStatistic> nodes,
                                 ImageType::Pointer& field_image);
    static void printMetric(std::vector<ReferenceROIStatistic> rois);
};

#endif // SPLINEINTERPOLATIONPROCESSOR_H
