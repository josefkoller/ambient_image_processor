#include "RegionGrowingSegmentationProcessor.h"

#include <itkIsolatedConnectedImageFilter.h>
#include <itkAddImageFilter.h>

RegionGrowingSegmentationProcessor::RegionGrowingSegmentationProcessor()
{

}


RegionGrowingSegmentationProcessor::LabelImage::Pointer RegionGrowingSegmentationProcessor::process(
    ITKImage gradient_image,
    std::vector<std::vector<Index> > input_segments,
    float tolerance)
{

    LabelImage::Pointer output_labels = LabelImage::New();
    output_labels->SetRegions(gradient_image.getPointer()->GetLargestPossibleRegion());
    output_labels->Allocate();
    output_labels->FillBuffer(0);

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
    LabelImage::Pointer output_labels, uint segment_index,
    Index index,
    float tolerance)
{
    if(! output_labels->GetLargestPossibleRegion().IsInside(index))
        return;

    if(output_labels->GetPixel(index) != 0)
        return; // already visited

    if(gradient_image.getPixel(index) < tolerance )
    {
        output_labels->SetPixel(index, segment_index);

        auto index2 = index; index2[0] = index[0] - 1;
        grow(gradient_image, output_labels, segment_index, index2, tolerance);

        auto index3 = index; index3[0] = index2[0] + 1;
        grow(gradient_image, output_labels, segment_index, index3, tolerance);

        auto index4 = index; index4[1] = index[1] + 1;
        grow(gradient_image, output_labels, segment_index, index4, tolerance);

        auto index5 = index; index5[1] = index[1] - 1;
        grow(gradient_image, output_labels, segment_index, index5, tolerance);

        auto index6 = index; index5[2] = index[2] - 1;
        grow(gradient_image, output_labels, segment_index, index6, tolerance);

        auto index7 = index; index7[2] = index[2] + 1;
        grow(gradient_image, output_labels, segment_index, index7, tolerance);

    }
}
