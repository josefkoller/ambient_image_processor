#include "RescaleIntensityProcessor.h"

#include "CudaImageOperationsProcessor.h"

RescaleIntensityProcessor::RescaleIntensityProcessor()
{
}

ITKImage RescaleIntensityProcessor::process(ITKImage image, ITKImage::PixelType from, ITKImage::PixelType to)
{
    ITKImage::PixelType minimum, maximum;
    image.minimumAndMaximum(minimum, maximum);

    return process(image, from, to, minimum, maximum);
}

ITKImage RescaleIntensityProcessor::process(ITKImage image, ITKImage::PixelType from, ITKImage::PixelType to,
                                            ITKImage mask)
{
    if(mask.isNull())
        return process(image, from, to);

    ITKImage::PixelType minimum, maximum;
    image.minimumAndMaximumInsideMask(minimum, maximum, mask);

    return process(image, from, to, minimum, maximum);
}

ITKImage RescaleIntensityProcessor::process(ITKImage image, ITKImage::PixelType from, ITKImage::PixelType to,
                                            ITKImage::PixelType minimum, ITKImage::PixelType maximum)
{
    auto range_before = maximum - minimum;
    auto range_after = to - from;

    auto scale = range_after / range_before;

    image = CudaImageOperationsProcessor::addConstant(image, -minimum);
    image = CudaImageOperationsProcessor::multiplyConstant(image, scale);

    return CudaImageOperationsProcessor::addConstant(image, from);
}
