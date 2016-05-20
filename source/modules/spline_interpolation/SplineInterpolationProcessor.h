#ifndef SPLINEINTERPOLATIONPROCESSOR_H
#define SPLINEINTERPOLATIONPROCESSOR_H

#include "ITKImage.h"

#include <QVector>

class SplineInterpolationProcessor
{
private:
    SplineInterpolationProcessor();
public:

    typedef ITKImage::Index Point;
    struct ReferenceROIStatistic
    {
        ITKImage::PixelType mean_value;
        Point point;
    };

    static ITKImage process(ITKImage image, uint spline_order,
                                 uint spline_levels, uint spline_control_points,
                                 std::vector<ReferenceROIStatistic> nodes,
                                 ITKImage& field_image);
    static void printMetric(std::vector<ReferenceROIStatistic> rois);

    static ReferenceROIStatistic calculateStatisticInROI(QVector<Point> roi, ITKImage image);
};

#endif // SPLINEINTERPOLATIONPROCESSOR_H
