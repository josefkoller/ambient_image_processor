#include "RegionGrowingSegmentationProcessor.h"

#include <itkIsolatedConnectedImageFilter.h>
#include <itkAddImageFilter.h>

RegionGrowingSegmentationProcessor::RegionGrowingSegmentationProcessor()
{

}


RegionGrowingSegmentationProcessor::LabelImage::Pointer RegionGrowingSegmentationProcessor::process(
    SourceImage::Pointer source_image,
    std::vector<std::vector<SourceImage::IndexType> > input_segments,
    float tolerance)
{
    /* DOES NOT WORK!!!

    typedef itk::RegionCompetitionImageFilter<SourceImage, LabelImage> RegionGrowingFilter;
    RegionGrowingFilter::Pointer region_growing_filter = RegionGrowingFilter::New();
    region_growing_filter->SetMaximumNumberOfIterations(maximum_number_of_iterations);
    region_growing_filter->SetInputLabels(input_labels);
    region_growing_filter->SetInput(source_image);
    region_growing_filter->Update();
    return region_growing_filter->GetOutput();
    */


    // http://www.itk.org/Doxygen/html/Examples_2Segmentation_2IsolatedConnectedImageFilter_8cxx-example.html#_a5
    // multiple seeds, but just positive and negative segments -> permutate
    // with bilateral

    // checking each input segment against the background only (0)
    // the first segment is the background segment_index == 0

    LabelImage::Pointer output_labels = LabelImage::New();
    output_labels->SetRegions(source_image->GetLargestPossibleRegion());
    output_labels->Allocate();
    output_labels->FillBuffer(0);

    typedef itk::AddImageFilter<LabelImage, LabelImage> CombineLabelsFilter;
    typedef itk::IsolatedConnectedImageFilter<SourceImage, LabelImage> SegmenationFilter;

    for(unsigned int segment_index = 1; segment_index <  input_segments.size(); segment_index++)
    {
        SegmenationFilter::Pointer filter = SegmenationFilter::New();
        filter->SetInput(source_image);
        filter->SetIsolatedValueTolerance(tolerance); // default is 1
        filter->SetReplaceValue(segment_index);
        for(SegmenationFilter::IndexType index : input_segments[segment_index])
            filter->AddSeed1(index);
        for(SegmenationFilter::IndexType index : input_segments[0])
            filter->AddSeed2(index);
        filter->Update();

        CombineLabelsFilter::Pointer combine_labels_filter = CombineLabelsFilter::New();
        combine_labels_filter->SetInput1(output_labels);
        combine_labels_filter->SetInput2(filter->GetOutput());
        combine_labels_filter->Update();
        output_labels = combine_labels_filter->GetOutput();

        //output_labels = ITKImageProcessor::clone<LabelImage>(combine_labels_filter->GetOutput());
    }
    return output_labels;
}
