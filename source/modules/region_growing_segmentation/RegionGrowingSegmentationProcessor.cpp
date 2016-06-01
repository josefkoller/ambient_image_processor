#include "RegionGrowingSegmentationProcessor.h"

#include <itkIsolatedConnectedImageFilter.h>
#include <itkAddImageFilter.h>

#include <sys/resource.h>

long RegionGrowingSegmentationProcessor::grow_counter = 0;

RegionGrowingSegmentationProcessor::RegionGrowingSegmentationProcessor()
{

}


RegionGrowingSegmentationProcessor::LabelImage RegionGrowingSegmentationProcessor::process(
        const ITKImage& gradient_image,
        Segments input_segments)
{
    ITKImage::PixelType* gradient_image_raw = gradient_image.cloneToPixelArray();
    auto size = ITKImage::Size(gradient_image.width,
                                         gradient_image.height,
                                         gradient_image.depth);
    LabelImage output_labels = gradient_image.cloneSameSizeWithZeros();
    LabelImage::PixelType* output_labels_raw = output_labels.cloneToPixelArray();

    if(output_labels_raw == nullptr)
        return output_labels;

    if(!setNeededStackSize())
        return output_labels;

    const uint max_recursion_depth = 1024;

    for(unsigned int segment_index = 0; segment_index <  input_segments.size(); segment_index++)
    {
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

            grow(gradient_image_raw, size, output_labels_raw, segment_index+1, seed_point.position,
                 seed_point.tolerance, 0, max_recursion_depth, max_recursion_depth_reached);
        }
    }
    output_labels = LabelImage(output_labels.width, output_labels.height, output_labels.depth,
                               output_labels_raw);

    delete output_labels_raw;
    delete gradient_image_raw;

    return output_labels;
}

void RegionGrowingSegmentationProcessor::grow(
        ITKImage::PixelType* gradient_image, ITKImage::Size size,
        LabelImage::PixelType* output_labels,
        uint segment_index,
        const ITKImage::Index& index,
        float tolerance,
        uint recursion_depth,
        uint max_recursion_depth,
        std::function<void(SeedPoint point)> max_recursion_depth_reached)
{
    grow_counter++;


    if(recursion_depth > max_recursion_depth) {
        max_recursion_depth_reached(SeedPoint(index, tolerance));
        return;
    }
    recursion_depth++;

    /*
    if(! output_labels->contains(index))
        return;

    if(output_labels->getPixel(index) > 1e-4) // != 0
        return; // already visited

    if(gradient_image.getPixel(index) < tolerance )
    {
    */
    ITKImage::setPixel(output_labels, size, index, segment_index);

    LabelImage::Index index2 = {index[0] - 1, index[1], index[2]};
    if(growCondition(gradient_image, size, output_labels, index2, tolerance))
        grow(gradient_image, size, output_labels, segment_index, index2, tolerance,
             recursion_depth, max_recursion_depth, max_recursion_depth_reached);

    LabelImage::Index index3 = {index[0] + 1, index[1], index[2]};
    if(growCondition(gradient_image, size, output_labels, index3, tolerance))
        grow(gradient_image, size, output_labels, segment_index, index3, tolerance,
             recursion_depth, max_recursion_depth, max_recursion_depth_reached);

    LabelImage::Index index4 = {index[0], index[1] + 1, index[2]};
    if(growCondition(gradient_image, size, output_labels, index4, tolerance))
        grow(gradient_image, size, output_labels, segment_index, index4, tolerance,
             recursion_depth, max_recursion_depth, max_recursion_depth_reached);

    LabelImage::Index index5 = {index[0], index[1] - 1, index[2]};
    if(growCondition(gradient_image, size, output_labels, index5, tolerance))
        grow(gradient_image, size, output_labels, segment_index, index5, tolerance,
             recursion_depth, max_recursion_depth, max_recursion_depth_reached);

    LabelImage::Index index6 = {index[0], index[1], index[2] - 1};
    if(growCondition(gradient_image, size, output_labels, index6, tolerance))
        grow(gradient_image, size, output_labels, segment_index, index6, tolerance,
             recursion_depth, max_recursion_depth, max_recursion_depth_reached);

    LabelImage::Index index7 = {index[0], index[1], index[2] + 1};
    if(growCondition(gradient_image, size, output_labels, index7, tolerance))
        grow(gradient_image, size, output_labels, segment_index, index7, tolerance,
             recursion_depth, max_recursion_depth, max_recursion_depth_reached);
}

bool RegionGrowingSegmentationProcessor::growCondition(
        ITKImage::PixelType* gradient_image, ITKImage::Size size,
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
            ITKImage::getPixel(gradient_image, size, index) < tolerance;
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
