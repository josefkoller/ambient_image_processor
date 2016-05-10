#ifndef SEGMENTSTOLABELIMAGECONVERTER_H
#define SEGMENTSTOLABELIMAGECONVERTER_H

#include "../ITKImageProcessor.h"
#include "RegionGrowingSegmentation.h"

class SegmentsToLabelImageConverter
{
private:
    SegmentsToLabelImageConverter();
public:

    typedef itk::Image<unsigned char, InputDimension> LabelImage;
    typedef std::vector<RegionGrowingSegmentation::Segment> SegmentVector;

    static LabelImage::Pointer convert(SegmentVector segments, LabelImage::SizeType size);
};

#endif // SEGMENTSTOLABELIMAGECONVERTER_H
