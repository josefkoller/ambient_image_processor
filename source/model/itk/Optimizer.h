#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "ITKCircleImage.h"

class Optimizer
{
public:
    Optimizer();
    ITKImage::Pointer run(ITKImage::Pointer input_image, uint iteration_count,
                          ITKImage::Pointer& best_field_image);
private:
    ITKImage::Pointer applyRandom(ITKImage::Pointer input_image,
                                  ITKImage::Pointer& field_image);
    static ITKImage::Pointer multiply(ITKImage::Pointer image1, ITKImage::Pointer image2);
    static float metric(ITKImage::Pointer image);
    static float tv(ITKImage::Pointer image);
    static float entropy(ITKImage::Pointer image);
};

#endif // OPTIMIZER_H
