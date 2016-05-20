#ifndef SEGMENTSTOLABELIMAGECONVERTER_H
#define SEGMENTSTOLABELIMAGECONVERTER_H

#include "RegionGrowingSegmentation.h"

#include "ITKImage.h"
#include "RegionGrowingSegmentationProcessor.h"

class SegmentsToLabelImageConverter
{
private:
    SegmentsToLabelImageConverter();
public:

    typedef RegionGrowingSegmentationProcessor::LabelImage LabelImage;
    typedef std::vector<RegionGrowingSegmentation::Segment> SegmentVector;

    static LabelImage convert(SegmentVector segments, LabelImage::InnerITKImage::SizeType size);
};

#endif // SEGMENTSTOLABELIMAGECONVERTER_H
