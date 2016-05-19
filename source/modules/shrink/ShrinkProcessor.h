#ifndef SHRINKPROCESSOR_H
#define SHRINKPROCESSOR_H

#include "ITKImage.h"

class ShrinkProcessor
{
private:
    ShrinkProcessor();
public:
    static ITKImage process(ITKImage image,
                                             unsigned int shrink_factor_x,
                                             unsigned int shrink_factor_y,
                                             unsigned int shrink_factor_z);
};

#endif // SHRINKPROCESSOR_H
