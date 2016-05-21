#ifndef REGIONGROWINGSEGMENTATIONPROCESSOR_H
#define REGIONGROWINGSEGMENTATIONPROCESSOR_H

#include "ITKImage.h"

class RegionGrowingSegmentationProcessor
{
private:
    RegionGrowingSegmentationProcessor();

public:

    typedef ITKImage::InnerITKImage::IndexType Index;

    typedef ITKImage LabelImage;

    static LabelImage process(
            const ITKImage& gradient_image,
            std::vector<std::vector<Index> > input_segments,
            float tolerance);

private:

    static void grow(const ITKImage& gradient_image,
        LabelImage* output_labels, uint segment_index, Index index,
        float tolerance);
};

#endif // REGIONGROWINGSEGMENTATIONPROCESSOR_H
