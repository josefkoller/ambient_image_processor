#include "LineProfileProcessor.h"

#include <itkLineConstIterator.h>
LineProfileProcessor::LineProfileProcessor()
{
}


void LineProfileProcessor::intensity_profile(ITKImage image,
                                             LineProfile::Point point1,
                                             LineProfile::Point point2,
                                             std::vector<double>& intensities,
                                             std::vector<double>& distances)
{
    if(!image.contains(point1) ||
       !image.contains(point2))
        return;

    typedef ITKImage::InnerITKImage ImageType;
    typedef ITKImage::PixelType PixelType;

    itk::LineConstIterator<ImageType> iterator(image.getPointer(), point1, point2);
    while(! iterator.IsAtEnd())
    {
        intensities.push_back(iterator.Get());

        ImageType::IndexType index = iterator.GetIndex();
        const int dx = point1[0] - index[0];
        const int dy = point1[1] - index[1];

        if(ImageType::ImageDimension == 3)
        {
            const int dz = point1[2] - index[2];
            distances.push_back(sqrt(dx*dx + dy*dy + dz*dz));
        }
        else
            distances.push_back(sqrt(dx*dx + dy*dy));

        ++iterator;
    }
}
