#include "ShrinkProcessor.h"

#include <itkShrinkImageFilter.h>

ShrinkProcessor::ShrinkProcessor()
{
}


ITKImage ShrinkProcessor::process(ITKImage image,
            unsigned int shrink_factor_x,
            unsigned int shrink_factor_y,
            unsigned int shrink_factor_z)
{
    if(image.isNull())
        return ITKImage();

    typedef ITKImage::InnerITKImage Image;

    typedef itk::ShrinkImageFilter<Image, Image> Shrinker;
    typename Shrinker::Pointer shrinker = Shrinker::New();
    shrinker->SetInput( image.getPointer() );
    shrinker->SetShrinkFactor(0, shrink_factor_x);
    shrinker->SetShrinkFactor(1, shrink_factor_y);
    shrinker->SetShrinkFactor(2, shrink_factor_z);

    shrinker->Update();
    Image::Pointer result = shrinker->GetOutput();
    result->DisconnectPipeline();

    return ITKImage(result);
}
