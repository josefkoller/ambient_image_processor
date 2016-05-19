#include "LineProfileProcessor.h"

#include <itkLineConstIterator.h>
LineProfileProcessor::LineProfileProcessor()
{
}


void LineProfileProcessor::intensity_profile(const ITKImage & image,
                                             int point1_x, int point1_y,
                                             int point2_x, int point2_y,
                                             std::vector<double>& intensities,
                                             std::vector<double>& distances)
{
    typedef ITKImage::InnerITKImage ImageType;
    typedef ITKImage::PixelType PixelType;

    ImageType::IndexType index1;
    index1[0] = point1_x;
    index1[1] = point1_y;


    ImageType::IndexType index2;
    index2[0] = point2_x;
    index2[1] = point2_y;

    if(ImageType::ImageDimension > 2) {
        index1[2] = 0;
        index2[2] = 0;
    }

    itk::LineConstIterator<ImageType> iterator(image.getPointer(), index1, index2);
    while(! iterator.IsAtEnd())
    {
        PixelType intensity = iterator.Get();
        ImageType::IndexType index = iterator.GetIndex();

        const int point_x = index[0];
        const int point_y = index[1];

        const int dx = point1_x - point_x;
        const int dy = point1_y - point_y;
        const double distance = sqrt(dx*dx + dy*dy);

        intensities.push_back(intensity);
        distances.push_back(distance);

        ++iterator;
    }
}
