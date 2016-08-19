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
        SegmentEdgePixelsMap segment_edge_pixels;

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

        // copy map keys to vector and transform the linear index...
        SegmentEdgePixelsVector segment_edge_pixels_vector;
        for(auto iterator = segment_edge_pixels.begin(); iterator != segment_edge_pixels.end(); ++iterator) {
            auto linear_index = iterator->first;
            segment_edge_pixels_vector.push_back(source_image.linearTo3DIndex(linear_index));
        }
        edge_pixels.push_back(segment_edge_pixels_vector);
    }

    output_labels = LabelImage(output_labels.width, output_labels.height, output_labels.depth,
                               output_labels_raw);

    output_labels.setOriginAndSpacingOf(source_image);

    delete[] output_labels_raw;
    delete[] source_image_raw;

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
        SegmentEdgePixelsMap& edge_pixels)
{
    grow_counter++;


    if(recursion_depth > max_recursion_depth) {
        max_recursion_depth_reached(SeedPoint(index, tolerance));
        return;
    }
    recursion_depth++;

    ITKImage::setPixel(output_labels, size, index, segment_index);

    bool is_edge_pixel = false;
    bool already_was_part_of_the_region = false;
    bool z_index_out_of_range = false;

    auto grow_step = [source_image, &size, output_labels, tolerance, segment_index, &is_edge_pixel,
            recursion_depth, max_recursion_depth, max_recursion_depth_reached,
            &edge_pixels, &already_was_part_of_the_region, &z_index_out_of_range] (
            const ITKImage::Index& grow_index){
        if(growCondition(source_image, size, output_labels, grow_index, tolerance,
                         already_was_part_of_the_region, z_index_out_of_range))
        {
            grow(source_image, size, output_labels, segment_index, grow_index, tolerance,
                 recursion_depth, max_recursion_depth, max_recursion_depth_reached, edge_pixels);
        }
        else if (!already_was_part_of_the_region && (size.z == 1 && !z_index_out_of_range))
        {
            is_edge_pixel = true;
        }
    };

    grow_step({index[0] - 1, index[1], index[2]});
    grow_step({index[0] + 1, index[1], index[2]});
    grow_step({index[0], index[1] + 1, index[2]});
    grow_step({index[0], index[1] - 1, index[2]});
    grow_step({index[0], index[1], index[2] - 1});
    grow_step({index[0], index[1], index[2] + 1});

    if(is_edge_pixel)
        edge_pixels[ITKImage::linearIndex(size, index)] = true;
}

bool RegionGrowingSegmentationProcessor::growCondition(
        ITKImage::PixelType* source_image, ITKImage::Size size,
        LabelImage::PixelType* output_labels,
        const ITKImage::InnerITKImage::IndexType& index,
        float tolerance,
        bool& already_was_part_of_the_region,
        bool& z_index_out_of_range)
{
    already_was_part_of_the_region = false;
    z_index_out_of_range = index[2] < 0 || index[2] >= size.z;
    if( index[0] >= 0 && index[0] < size.x &&
        index[1] >= 0 && index[1] < size.y &&
        !z_index_out_of_range)
    {
        already_was_part_of_the_region = ITKImage::getPixel(output_labels, size, index) > 1e-4;
        if(!already_was_part_of_the_region)
            return ITKImage::getPixel(source_image, size, index) < tolerance;
    }
    return false;
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
