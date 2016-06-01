#ifndef REGIONGROWINGSEGMENTATIONPROCESSOR_H
#define REGIONGROWINGSEGMENTATIONPROCESSOR_H

#include "ITKImage.h"

#include <functional>

#include "RegionGrowingSegmentation.h"

class RegionGrowingSegmentationProcessor
{
private:
    RegionGrowingSegmentationProcessor();

    typedef RegionGrowingSegmentation::Segments Segments;
    typedef RegionGrowingSegmentation::SeedPoint SeedPoint;
public:

    typedef ITKImage::InnerITKImage::IndexType Index;

    typedef ITKImage LabelImage;

    static LabelImage process(
            const ITKImage& gradient_image,
            Segments input_segments);

private:

    static long grow_counter;

    static void grow(ITKImage::PixelType* gradient_image, ITKImage::Size size,
                     LabelImage::PixelType* output_labels,
                     uint segment_index,
                     const ITKImage::Index& index,
                     float tolerance,
                     uint recursion_depth,
                     uint max_recursion_depth,
                     std::function<void(SeedPoint point)> max_recursion_depth_reached);
    static bool growCondition(ITKImage::PixelType* gradient_image, ITKImage::Size size,
                              LabelImage::PixelType* output_labels,
                              const Index& index,
                              float tolerance);

    static bool setNeededStackSize();
};

#endif // REGIONGROWINGSEGMENTATIONPROCESSOR_H
