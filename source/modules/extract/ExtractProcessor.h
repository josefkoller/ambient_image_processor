#ifndef EXTRACTPROCESSOR_H
#define EXTRACTPROCESSOR_H

#include "ITKImage.h"

class ExtractProcessor
{
private:
    ExtractProcessor();
public:
    static ITKImage process(ITKImage image,
         unsigned int from_x, unsigned int to_x,
         unsigned int from_y, unsigned int to_y,
         unsigned int from_z, unsigned int to_z);

    static ITKImage process(ITKImage image, ITKImage::InnerITKImage::RegionType region);

    static ITKImage extract_slice(ITKImage image,
         unsigned int slice_index);
};

#endif // EXTRACTPROCESSOR_H
