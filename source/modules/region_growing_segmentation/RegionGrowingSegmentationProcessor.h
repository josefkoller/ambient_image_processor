/*
    Ambient Image Processor - A tool to perform several imaging tasks
    
    Copyright (C) 2016 Josef Koller

    https://github.com/josefkoller/ambient_image_processor    
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

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
