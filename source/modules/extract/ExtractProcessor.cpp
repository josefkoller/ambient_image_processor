#include "ExtractProcessor.h"

#include <itkImageRegionConstIteratorWithIndex.h>

ExtractProcessor::ExtractProcessor()
{
}

ITKImage ExtractProcessor::process(
     ITKImage itk_image,
     unsigned int from_x, unsigned int to_x,
     unsigned int from_y, unsigned int to_y,
     unsigned int from_z, unsigned int to_z)
{
    typedef ITKImage::InnerITKImage ImageType;
    ImageType::Pointer image = itk_image.getPointer();

    ImageType::SizeType extract_size;
    extract_size[0] = to_x - from_x + 1;
    extract_size[1] = to_y - from_y + 1;
    bool is3D = image->GetLargestPossibleRegion().GetSize().Dimension > 2;
    if(is3D)
        extract_size[2] = to_z - from_z + 1;
    ImageType::RegionType extract_region(extract_size);

    return process(image, extract_region);
}

ITKImage ExtractProcessor::process(ITKImage image,
                                          ITKImage::InnerITKImage::RegionType extract_region)
{
    typedef ITKImage::InnerITKImage ImageType;
    ImageType::Pointer extracted_volume = ImageType::New();
    extracted_volume->SetRegions(extract_region);
    extracted_volume->Allocate();
    extracted_volume->SetSpacing(image.getPointer()->GetSpacing());

    ImageType::IndexType start_index = extract_region.GetIndex();

    itk::ImageRegionConstIteratorWithIndex<ImageType> source_iteration(
                image.getPointer(), extract_region);
    while(!source_iteration.IsAtEnd())
    {
        ImageType::IndexType index = source_iteration.GetIndex();
        ImageType::IndexType destination_index;
        destination_index[0] = index[0] + start_index[0];
        destination_index[1] = index[1] + start_index[1];
        destination_index[2] = index[2] + start_index[2];

        extracted_volume->SetPixel(destination_index, source_iteration.Get());

        ++source_iteration;
    }

    ImageType::SpacingType spacing = image.getPointer()->GetSpacing();
    ImageType::PointType origin = image.getPointer()->GetOrigin();
    origin[0] = origin[0] + start_index[0] * spacing[0];
    origin[1] = origin[1] + start_index[1] * spacing[1];
    origin[2] = origin[2] + start_index[2] * spacing[2];
    extracted_volume->SetOrigin(origin);

    return ITKImage(extracted_volume);
}

ITKImage ExtractProcessor::extract_slice(ITKImage image,
     unsigned int slice_index)
{
    uint to_x = image.width - 1;
    uint to_y = image.height - 1;

    return process(image, 0, to_x, 0, to_y, slice_index, slice_index);
}
