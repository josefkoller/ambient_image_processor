/*
    Ambient Image Processor - A tool to perform several imaging tasks
    
    Copyright (C) 2016 Josef Koller

    https://github.com/josefkoller/ambient_image_processor    
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "ExtractProcessor.h"

#include <itkImageRegionConstIteratorWithIndex.h>

ExtractProcessor::ExtractProcessor()
{
}

ITKImage ExtractProcessor::process(
        ITKImage image,
        unsigned int from_x, unsigned int to_x,
        unsigned int from_y, unsigned int to_y,
        unsigned int from_z, unsigned int to_z)
{
    ITKImage::Index start;
    start[0] = from_x;
    start[1] = from_y;
    start[2] = from_z;

    ITKImage::Index end;
    end[0] = to_x;
    end[1] = to_y;
    end[2] = to_z;

    ITKImage::InnerITKImage::RegionType extract_region;
    extract_region.SetIndex(start);
    extract_region.SetUpperIndex(end);

    ITKImage::InnerITKImage::SizeType size = image.getPointer()->GetLargestPossibleRegion().GetSize();
    std::ostringstream message;
    message << "extracting from image size " << size <<  ": " << start << " - " << end;
    // std::cout << message.str() << std::endl;

    if(from_x > size[0] - 1  || from_x < 0 ||
            from_y > size[1] - 1 || from_y < 0 ||
            from_z > size[2] - 1 || from_z < 0)
        throw std::runtime_error("invalid extraction parameters: " + message.str());

    return process(image, extract_region);
}

ITKImage ExtractProcessor::process(ITKImage image,
                                   ITKImage::InnerITKImage::RegionType extract_region)
{
    typedef ITKImage::InnerITKImage ImageType;
    ImageType::Pointer extracted_volume = ImageType::New();

    extracted_volume->SetRegions(extract_region.GetSize());
    extracted_volume->Allocate();
    extracted_volume->SetSpacing(image.getPointer()->GetSpacing());

    ImageType::IndexType start_index = extract_region.GetIndex();

    itk::ImageRegionConstIteratorWithIndex<ImageType> source_iteration(
                image.getPointer(), extract_region);
    while(!source_iteration.IsAtEnd())
    {
        ImageType::IndexType index = source_iteration.GetIndex();
        ImageType::IndexType destination_index;
        destination_index[0] = index[0] - start_index[0];
        destination_index[1] = index[1] - start_index[1];
        destination_index[2] = index[2] - start_index[2];

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
