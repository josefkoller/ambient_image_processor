#include "SegmentsToLabelImageConverter.h"

SegmentsToLabelImageConverter::SegmentsToLabelImageConverter()
{

}


SegmentsToLabelImageConverter::LabelImage
    SegmentsToLabelImageConverter::convert(SegmentVector segments, LabelImage::InnerITKImage::SizeType size)
{
    LabelImage label_image = LabelImage(size[0], size[1], size[2]);
    label_image.setEachPixel([](uint,uint,uint) {
        return 0;
    });

    LabelImage::PixelType segment_pixel_value = 1;
    for(RegionGrowingSegmentation::Segment segment : segments)
    {
        for(auto seed_point : segment.seed_points)
        {
            label_image.setPixel(seed_point.position, segment_pixel_value);
        }
        segment_pixel_value++;
    }
    return label_image;
}
