#include "CudaImageOperationsProcessor.h"

template<typename Pixel>
Pixel* multiply_kernel_launch(Pixel* image1, Pixel* image2,
                              uint width, uint height, uint depth);

CudaImageOperationsProcessor::CudaImageOperationsProcessor()
{
}


ITKImage CudaImageOperationsProcessor::multiply(
        ITKImage image1, ITKImage image2)
{
    auto image1_pixels = image1.cloneToPixelArray();
    auto image2_pixels = image2.cloneToPixelArray();

    auto result_pixels = multiply_kernel_launch(image1_pixels, image2_pixels,
                                                image1.width,
                                                image1.height,
                                                image1.depth);


    auto result = ITKImage(image1.width, image1.height, image1.depth, result_pixels);

    delete[] image1_pixels;
    delete[] image2_pixels;
    delete[] result_pixels;

    return result;
}
