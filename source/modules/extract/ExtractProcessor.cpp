#include "ExtractProcessor.h"


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

    ImageType::Pointer extracted_volume = ImageType::New();
    extracted_volume->SetRegions(extract_region);
    extracted_volume->Allocate();
    extracted_volume->SetSpacing(image->GetSpacing());

    for(unsigned int x = from_x; x <= to_x; x++)
    {
        for(unsigned int y = from_y; y <= to_y; y++)
        {
            for(unsigned int z = from_z; z <= to_z; z++)
            {
                ImageType::IndexType index_in_source;
                index_in_source[0] = x;
                index_in_source[1] = y;
                if(is3D)
                    index_in_source[2] = z;
                ImageType::PixelType pixel = image->GetPixel(index_in_source);

                ImageType::IndexType index_in_extracted;
                index_in_extracted[0] = x - from_x;
                index_in_extracted[1] = y - from_y;
                if(is3D)
                    index_in_extracted[2] = z - from_z;
                extracted_volume->SetPixel(index_in_extracted, pixel);
            }
        }
    }

    ImageType::SpacingType spacing = image->GetSpacing();
    ImageType::PointType origin = image->GetOrigin();
    origin[0] = origin[0] + from_x * spacing[0];
    origin[1] = origin[1] + from_y * spacing[1];
    if(is3D)
        origin[2] = origin[2] + from_z * spacing[2];
    extracted_volume->SetOrigin(origin);

    return ITKImage(extracted_volume);
}
