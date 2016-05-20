#include "RegionGrowingSegmentationProcessor.h"

#include <itkIsolatedConnectedImageFilter.h>
#include <itkAddImageFilter.h>

RegionGrowingSegmentationProcessor::RegionGrowingSegmentationProcessor()
{

}


RegionGrowingSegmentationProcessor::LabelImage RegionGrowingSegmentationProcessor::process(
    const ITKImage& gradient_image,
    std::vector<std::vector<Index> > input_segments,
    float tolerance)
{

    LabelImage output_labels = gradient_image.cloneSameSizeWithZeros();

    for(unsigned int segment_index = 0; segment_index <  input_segments.size(); segment_index++)
    {
        std::vector<Index> seed_points = input_segments[segment_index];

        for(Index seed_point : seed_points)
        {
            grow(gradient_image, output_labels, segment_index+1, seed_point, tolerance);
        }
    }
    return output_labels;
}

void RegionGrowingSegmentationProcessor::grow(
    const ITKImage& gradient_image,
    LabelImage& output_labels, uint segment_index,
    Index index,
    float tolerance)
{
    if(! output_labels.contains(index))
        return;

    if(output_labels.getPixel(index) > 1e-4) // != 0
        return; // already visited

    if(gradient_image.getPixel(index) < tolerance )
    {
        output_labels.setPixel(index, segment_index);

        LabelImage::Index index2 = index; index2[0] = index[0] - 1;
        if(output_labels.contains(index2))
            grow(gradient_image, output_labels, segment_index, index2, tolerance);

        LabelImage::Index index3 = index; index3[0] = index[0] + 1;
        if(output_labels.contains(index3))
            grow(gradient_image, output_labels, segment_index, index3, tolerance);

        LabelImage::Index index4 = index; index4[1] = index[1] + 1;
        if(output_labels.contains(index4))
            grow(gradient_image, output_labels, segment_index, index4, tolerance);

        LabelImage::Index index5 = index; index5[1] = index[1] - 1;
        if(output_labels.contains(index5))
            grow(gradient_image, output_labels, segment_index, index5, tolerance);

        LabelImage::Index index6 = index; index6[2] = index[2] - 1;
        if(output_labels.contains(index6))
            grow(gradient_image, output_labels, segment_index, index6, tolerance);

        LabelImage::Index index7 = index; index7[2] = index[2] + 1;
        if(output_labels.contains(index7))
            grow(gradient_image, output_labels, segment_index, index7, tolerance);

    }
}
