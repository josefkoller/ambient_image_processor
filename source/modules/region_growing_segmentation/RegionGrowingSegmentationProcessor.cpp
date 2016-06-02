#include "RegionGrowingSegmentationProcessor.h"

#include <itkIsolatedConnectedImageFilter.h>
#include <itkAddImageFilter.h>

#include <sys/resource.h>

long RegionGrowingSegmentationProcessor::grow_counter = 0;

RegionGrowingSegmentationProcessor::RegionGrowingSegmentationProcessor()
{

}


RegionGrowingSegmentationProcessor::LabelImage RegionGrowingSegmentationProcessor::process(
        const ITKImage& source_image,
        Segments input_segments,
        EdgePixelsCollection& edge_pixels)
{
    ITKImage::PixelType* source_image_raw = source_image.cloneToPixelArray();
    auto size = ITKImage::Size(source_image.width,
                                         source_image.height,
                                         source_image.depth);
    LabelImage output_labels = source_image.cloneSameSizeWithZeros();
    LabelImage::PixelType* output_labels_raw = output_labels.cloneToPixelArray();

    if(output_labels_raw == nullptr)
        return output_labels;

    if(!setNeededStackSize())
        return output_labels;

    const uint max_recursion_depth = 1024;

    for(unsigned int segment_index = 0; segment_index <  input_segments.size(); segment_index++)
    {
        SegmentEdgePixelsCollection segment_edge_pixels;

        auto seed_points = input_segments[segment_index].seed_points;

        std::queue<SeedPoint> seed_point_queue;
        for(auto seed_point : seed_points)
            seed_point_queue.push(seed_point);

        auto max_recursion_depth_reached = [&seed_point_queue] (SeedPoint point) {
            seed_point_queue.push(point);
        };

        while(!seed_point_queue.empty()) {
            auto seed_point = seed_point_queue.front();
            seed_point_queue.pop();

            grow(source_image_raw, size, output_labels_raw, segment_index+1, seed_point.position,
                 seed_point.tolerance, 0, max_recursion_depth, max_recursion_depth_reached,
                 segment_edge_pixels);
        }

        edge_pixels.push_back(segment_edge_pixels);
    }

    output_labels = LabelImage(output_labels.width, output_labels.height, output_labels.depth,
                               output_labels_raw);

    delete output_labels_raw;
    delete source_image_raw;

    return output_labels;
}

void RegionGrowingSegmentationProcessor::grow(
        ITKImage::PixelType* source_image, ITKImage::Size size,
        LabelImage::PixelType* output_labels,
        uint segment_index,
        const ITKImage::Index& index,
        float tolerance,
        uint recursion_depth,
        uint max_recursion_depth,
        std::function<void(SeedPoint point)> max_recursion_depth_reached,
        SegmentEdgePixelsCollection& edge_pixels)
{
    grow_counter++;


    if(recursion_depth > max_recursion_depth) {
        max_recursion_depth_reached(SeedPoint(index, tolerance));
        return;
    }
    recursion_depth++;

    ITKImage::setPixel(output_labels, size, index, segment_index);

    bool is_edge_pixel = false;
    LabelImage::Index index2 = {index[0] - 1, index[1], index[2]};
    if(growCondition(source_image, size, output_labels, index2, tolerance))
        grow(source_image, size, output_labels, segment_index, index2, tolerance,
             recursion_depth, max_recursion_depth, max_recursion_depth_reached, edge_pixels);
    else
        is_edge_pixel = true;

    LabelImage::Index index3 = {index[0] + 1, index[1], index[2]};
    if(growCondition(source_image, size, output_labels, index3, tolerance))
        grow(source_image, size, output_labels, segment_index, index3, tolerance,
             recursion_depth, max_recursion_depth, max_recursion_depth_reached, edge_pixels);
    else
        is_edge_pixel = true;

    LabelImage::Index index4 = {index[0], index[1] + 1, index[2]};
    if(growCondition(source_image, size, output_labels, index4, tolerance))
        grow(source_image, size, output_labels, segment_index, index4, tolerance,
             recursion_depth, max_recursion_depth, max_recursion_depth_reached, edge_pixels);
    else
        is_edge_pixel = true;

    LabelImage::Index index5 = {index[0], index[1] - 1, index[2]};
    if(growCondition(source_image, size, output_labels, index5, tolerance))
        grow(source_image, size, output_labels, segment_index, index5, tolerance,
             recursion_depth, max_recursion_depth, max_recursion_depth_reached, edge_pixels);
    else
        is_edge_pixel = true;

    LabelImage::Index index6 = {index[0], index[1], index[2] - 1};
    if(growCondition(source_image, size, output_labels, index6, tolerance))
        grow(source_image, size, output_labels, segment_index, index6, tolerance,
             recursion_depth, max_recursion_depth, max_recursion_depth_reached, edge_pixels);
    else
        is_edge_pixel = true;

    LabelImage::Index index7 = {index[0], index[1], index[2] + 1};
    if(growCondition(source_image, size, output_labels, index7, tolerance))
        grow(source_image, size, output_labels, segment_index, index7, tolerance,
             recursion_depth, max_recursion_depth, max_recursion_depth_reached, edge_pixels);
    else
        is_edge_pixel = true;

    if(is_edge_pixel)
        edge_pixels[ITKImage::linearIndex(size, index)] = true;
}

bool RegionGrowingSegmentationProcessor::growCondition(
        ITKImage::PixelType* source_image, ITKImage::Size size,
        LabelImage::PixelType* output_labels,
        const ITKImage::InnerITKImage::IndexType& index,
        float tolerance)
{
    return
            //index >= ITKImage::PixelIndex() && index < size
            index[0] >= 0 && index[0] < size.x &&
            index[1] >= 0 && index[1] < size.y &&
            index[2] >= 0 && index[2] < size.z &&
            ITKImage::getPixel(output_labels, size, index) < 1e-4 &&
            ITKImage::getPixel(source_image, size, index) < tolerance;
}

bool RegionGrowingSegmentationProcessor::setNeededStackSize()
{
    const rlim_t needed_stack_size = 16777216 * 2; // bytes
    struct rlimit limit;
    getrlimit(RLIMIT_STACK, &limit);
    if(limit.rlim_cur < needed_stack_size) {
        if(limit.rlim_max > needed_stack_size) {
            limit.rlim_cur = needed_stack_size;
            int result = setrlimit(RLIMIT_STACK, &limit);
            if(result != 0) {
                std::cerr << "could not set the stack size to " << needed_stack_size << std::endl;
                return false;
            }

            struct rlimit limit2;
            getrlimit(RLIMIT_STACK, &limit2);
            if(limit2.rlim_cur != needed_stack_size) {
                std::cerr << "did not set the stack size to " << needed_stack_size << std::endl;
                return false;
            }

            return true;
        }
        std::cerr << "maximum stack size is smaller than " << needed_stack_size << std::endl;
        return false;
    }
    return true;
}
