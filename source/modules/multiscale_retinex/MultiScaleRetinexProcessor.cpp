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

#include "MultiScaleRetinexProcessor.h"

#include <itkDiscreteGaussianImageFilter.h>
#include <itkLogImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkMultiplyImageFilter.h>

MultiScaleRetinexProcessor::MultiScaleRetinexProcessor()
{
}


ITKImage MultiScaleRetinexProcessor::process(
        ITKImage image,
        std::vector<MultiScaleRetinex::Scale*> scales)
{
    typedef ITKImage::InnerITKImage ImageType;

    //normalize weights
    ImageType::PixelType weight_sum = 0;
    for(MultiScaleRetinex::Scale* scale : scales)
        weight_sum += scale->weight;

    ImageType::Pointer input_image =  image.clone().getPointer();
    ImageType::RegionType region = input_image->GetLargestPossibleRegion();

    ImageType::PointType origin;
    origin.Fill(0);
    ImageType::SpacingType spacing;
    spacing.Fill(1);
    input_image->SetOrigin(origin);
    input_image->SetSpacing(spacing);

    ImageType::Pointer reflectance = ImageType::New();
    reflectance->SetRegions(region);
    reflectance->Allocate();
    reflectance->FillBuffer(0);

    typedef itk::DiscreteGaussianImageFilter<ImageType, ImageType> GaussFilter;
    typedef itk::LogImageFilter<ImageType, ImageType> LogFilter;
    typedef itk::SubtractImageFilter<ImageType, ImageType, ImageType> SubtractFilter;
    typedef itk::AddImageFilter<ImageType, ImageType, ImageType> AddFilter;
    typedef itk::MultiplyImageFilter<ImageType, ImageType, ImageType> MultiplyFilter;

    AddFilter::Pointer add_filter_input = AddFilter::New();
    add_filter_input->SetInput1(input_image);
    add_filter_input->SetConstant2(1);
    add_filter_input->Update();

    LogFilter::Pointer log_filter_input = LogFilter::New();
    log_filter_input->SetInput(add_filter_input->GetOutput());
    log_filter_input->Update();

    for(MultiScaleRetinex::Scale* scale : scales)
    {
        GaussFilter::Pointer gauss_filter = GaussFilter::New();
        gauss_filter->SetInput(input_image);
        gauss_filter->SetMaximumKernelWidth(32);
        gauss_filter->SetVariance(scale->sigma);
        gauss_filter->Update();

        AddFilter::Pointer add_filter1 = AddFilter::New();
        add_filter1->SetInput1(gauss_filter->GetOutput());
        add_filter1->SetConstant2(1);
        add_filter1->Update();

        LogFilter::Pointer log_filter1 = LogFilter::New();
        log_filter1->SetInput(add_filter1->GetOutput());
        log_filter1->Update();

        SubtractFilter::Pointer subtract_filter = SubtractFilter::New();
        subtract_filter->SetInput1(log_filter_input->GetOutput());
        subtract_filter->SetInput2(log_filter1->GetOutput());
        subtract_filter->Update();

        MultiplyFilter::Pointer multiply_filter = MultiplyFilter::New();
        multiply_filter->SetInput1(subtract_filter->GetOutput());
        multiply_filter->SetConstant2(scale->weight / weight_sum);
        multiply_filter->Update();

        AddFilter::Pointer add_filter = AddFilter::New();
        add_filter->SetInput1(reflectance);
        add_filter->SetInput2(multiply_filter->GetOutput());
        add_filter->Update();
        reflectance = add_filter->GetOutput();

    }

    origin = input_image.GetPointer()->GetOrigin();
    spacing = input_image.GetPointer()->GetSpacing();
    reflectance->SetOrigin(origin);
    reflectance->SetSpacing(spacing);

    return ITKImage(reflectance);
}
