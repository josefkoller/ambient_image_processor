#ifndef REGIONGROWINGSEGMENTATIONPROCESSOR_H
#define REGIONGROWINGSEGMENTATIONPROCESSOR_H

#include "ITKImage.h"

#include <functional>

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
    static long grow_counter;

    static void grow(ITKImage::PixelType* gradient_image, ITKImage::Size size,
                     LabelImage::PixelType* output_labels,
                     uint segment_index,
                     const ITKImage::PixelIndex& index,
                     float tolerance,
                     uint recursion_depth,
                     uint max_recursion_depth,
                     std::function<void(ITKImage::PixelIndex index)> max_recursion_depth_reached);
    static bool growCondition(ITKImage::PixelType* gradient_image, ITKImage::Size size,
                              LabelImage::PixelType* output_labels,
                              const ITKImage::PixelIndex& index,
                              float tolerance);

    static bool setNeededStackSize();
};

#endif // REGIONGROWINGSEGMENTATIONPROCESSOR_H
