#ifndef REGIONGROWINGSEGMENTATIONPROCESSOR_H
#define REGIONGROWINGSEGMENTATIONPROCESSOR_H

#include "ITKImage.h"

#include <functional>
#include <unordered_map>

#include "RegionGrowingSegmentation.h"

class RegionGrowingSegmentationProcessor
{
private:
    RegionGrowingSegmentationProcessor();

    typedef RegionGrowingSegmentation::Segments Segments;
    typedef RegionGrowingSegmentation::SeedPoint SeedPoint;
public:

    typedef ITKImage::Index Index;

    typedef ITKImage LabelImage;

    typedef std::unordered_map<uint, bool> SegmentEdgePixelsCollection;
    typedef std::vector<SegmentEdgePixelsCollection> EdgePixelsCollection;

    static LabelImage process(
            const ITKImage& source_image,
            Segments input_segments,
            EdgePixelsCollection& edge_pixels);

private:

    static long grow_counter;

    static void grow(ITKImage::PixelType* source_image, ITKImage::Size size,
                     LabelImage::PixelType* output_labels,
                     uint segment_index,
                     const ITKImage::Index& index,
                     float tolerance,
                     uint recursion_depth,
                     uint max_recursion_depth,
                     std::function<void(SeedPoint point)> max_recursion_depth_reached,
                     SegmentEdgePixelsCollection& edge_pixels);
    static bool growCondition(ITKImage::PixelType* source_image, ITKImage::Size size,
                              LabelImage::PixelType* output_labels,
                              const Index& index,
                              float tolerance);

    static bool setNeededStackSize();
};

#endif // REGIONGROWINGSEGMENTATIONPROCESSOR_H
