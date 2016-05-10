#include "SegmentsToLabelImageConverter.h"

SegmentsToLabelImageConverter::SegmentsToLabelImageConverter()
{

}


static SegmentsToLabelImageConverter::LabelImage::Pointer
    SegmentsToLabelImageConverter::convert(SegmentVector segments, LabelImage::SizeType size)
{
    LabelImage::Pointer label_image = LabelImage::New();
    label_image->SetRegions(size);
    label_image->Allocate();
    // 0 for pixels which are not in any segment...
    label_image->FillBuffer(0);

    LabelImage::PixelType segment_pixel_value = 1;
    for(RegionGrowingSegmentation::Segment segment : segments)
    {
        for(RegionGrowingSegmentation::Position index : segment.seed_points)
        {
            label_image->SetPixel(index, segment_pixel_value);
        }
        segment_pixel_value++;
    }
    return label_image;
}
