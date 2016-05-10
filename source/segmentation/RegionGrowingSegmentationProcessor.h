#ifndef REGIONGROWINGSEGMENTATIONPROCESSOR_H
#define REGIONGROWINGSEGMENTATIONPROCESSOR_H

#include "../ITKImageProcessor.h"

class RegionGrowingSegmentationProcessor
{
private:
    RegionGrowingSegmentationProcessor();
public:

    typedef ITKImageProcessor::ImageType SourceImage;
    typedef itk::Image<unsigned char, InputDimension> LabelImage;

    static LabelImage::Pointer process(
            SourceImage::Pointer source_image,
            std::vector<std::vector<SourceImage::IndexType> > input_segments,
            float tolerance);
};

#endif // REGIONGROWINGSEGMENTATIONPROCESSOR_H
