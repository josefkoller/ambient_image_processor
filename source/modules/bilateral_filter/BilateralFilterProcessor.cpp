#include "BilateralFilterProcessor.h"

#include <itkBilateralImageFilter.h>

BilateralFilterProcessor::BilateralFilterProcessor()
{
}



ITKImage BilateralFilterProcessor::process(ITKImage image,
                                          ImageType::PixelType sigma_spatial_distance,
                                          ImageType::PixelType sigma_intensity_distance,
                                          int kernel_size)
{
    typedef itk::BilateralImageFilter<ImageType,ImageType> BilateralFilter;
    BilateralFilter::Pointer filter = BilateralFilter::New();
    filter->SetInput(image.getPointer());
    filter->SetDomainSigma(sigma_intensity_distance);
    filter->SetRangeSigma(sigma_spatial_distance);
    filter->SetAutomaticKernelSize(false);
    BilateralFilter::SizeType radius;
    radius.Fill(kernel_size);
    filter->SetRadius(radius);

    filter->Update();
    ImageType::Pointer output = filter->GetOutput();
    output->DisconnectPipeline();

    return ITKImage(output);

    /*
     *default values:
          this->m_Radius.Fill(1);
          this->m_AutomaticKernelSize = true;
          this->m_DomainSigma.Fill(4.0);
          this->m_RangeSigma = 50.0;
          this->m_FilterDimensionality = ImageDimension;
          this->m_NumberOfRangeGaussianSamples = 100;
          this->m_DynamicRange = 0.0;
          this->m_DynamicRangeUsed = 0.0;
          this->m_DomainMu = 2.5;  // keep small to keep kernels small
          this->m_RangeMu = 4.0;
    */
}
