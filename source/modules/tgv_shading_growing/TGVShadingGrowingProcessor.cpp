#include "TGVShadingGrowingProcessor.h"

#include "NonLocalGradientProcessor.h"

template<typename Pixel>
Pixel* tgv2_l1_shading_growing(Pixel* f_host,
                               uint width, uint height, uint depth,
                               Pixel lambda,
                               uint iteration_count,
                               uint paint_iteration_interval,
                               TGVProcessor::IterationCallback<Pixel> iteration_finished_callback,
                               Pixel alpha0,
                               Pixel alpha1,
                               Pixel lower_threshold,
                               Pixel upper_threshold,
                               Pixel* non_local_gradient_kernel_host,
                               uint non_local_gradient_kernel_size);

TGVShadingGrowingProcessor::TGVShadingGrowingProcessor()
{
}


ITKImage TGVShadingGrowingProcessor::process(ITKImage input_image,
                                             const ITKImage::PixelType lambda,
                                             const ITKImage::PixelType alpha0,
                                             const ITKImage::PixelType alpha1,
                                             const uint iteration_count,
                                             const uint paint_iteration_interval,
                                             IterationFinished iteration_finished_callback,
                                             ITKImage::PixelType lower_threshold,
                                             ITKImage::PixelType upper_threshold,
                                             ITKImage::PixelType non_local_gradient_kernel_sigma,
                                             uint non_local_gradient_kernel_size)
{
    Pixel* f = input_image.cloneToPixelArray();

    IterationCallback<Pixel> iteration_callback = [&input_image, iteration_finished_callback] (
            uint iteration_index, uint iteration_count, Pixel* u) {
        auto itk_u = ITKImage(input_image.width, input_image.height, input_image.depth, u);
        return iteration_finished_callback(iteration_index, iteration_count, itk_u);
    };

    Pixel* non_local_gradient_kernel = NonLocalGradientProcessor::createKernel(non_local_gradient_kernel_size,
                                                   non_local_gradient_kernel_sigma).cloneToPixelArray();

    Pixel* u = tgv2_l1_shading_growing(f,
                                       input_image.width, input_image.height, input_image.depth,
                                       lambda,
                                       iteration_count, paint_iteration_interval, iteration_callback,
                                       alpha0, alpha1,
                                       lower_threshold, upper_threshold,
                                       non_local_gradient_kernel,
                                       non_local_gradient_kernel_size);
    delete[] non_local_gradient_kernel;
    delete[] f;

    auto result = ITKImage(input_image.width, input_image.height, input_image.depth, u);
    delete[] u;
    return result;
}
