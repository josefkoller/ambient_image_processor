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
public:

    typedef RegionGrowingSegmentation::Segments Segments;
    typedef RegionGrowingSegmentation::SeedPoint SeedPoint;

    typedef ITKImage::Index Index;

    typedef ITKImage LabelImage;

    typedef std::unordered_map<uint, bool> SegmentEdgePixelsMap;
    typedef std::vector<Index> SegmentEdgePixelsVector;
    typedef std::vector<SegmentEdgePixelsVector> EdgePixelsCollection;

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
                     SegmentEdgePixelsMap& edge_pixels);
    static bool growCondition(ITKImage::PixelType* source_image, ITKImage::Size size,
                              LabelImage::PixelType* output_labels,
                              const Index& index,
                              float tolerance,
                              bool& already_was_part_of_the_region,
                              bool& z_index_out_of_range);

    static bool setNeededStackSize();
};

#endif // REGIONGROWINGSEGMENTATIONPROCESSOR_H
