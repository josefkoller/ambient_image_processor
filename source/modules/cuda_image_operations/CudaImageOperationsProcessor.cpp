#include "CudaImageOperationsProcessor.h"

template<typename Pixel>
Pixel* multiply_kernel_launch(Pixel* image1, Pixel* image2,
                              uint width, uint height, uint depth);
template<typename Pixel>
Pixel* divide_kernel_launch(Pixel* image1, Pixel* image2,
                              uint width, uint height, uint depth);
template<typename Pixel>
Pixel* add_kernel_launch(Pixel* image1, Pixel* image2,
                              uint width, uint height, uint depth);
template<typename Pixel>
Pixel* convolution3x3_kernel_launch(Pixel* image1,
                              uint width, uint height, uint depth,
                                    Pixel* kernel);
template<typename Pixel>
Pixel* convolution3x3x3_kernel_launch(Pixel* image1,
                              uint width, uint height, uint depth,
                                    Pixel* kernel, bool correct_center);

CudaImageOperationsProcessor::CudaImageOperationsProcessor()
{
}

ITKImage CudaImageOperationsProcessor::multiply(ITKImage image1, ITKImage image2)
{
    return perform(image1, image2, [&image1](Pixels pixels1, Pixels pixels2) {
        return multiply_kernel_launch(pixels1, pixels2,
                                      image1.width,
                                      image1.height,
                                      image1.depth);
    });
}

ITKImage CudaImageOperationsProcessor::divide(ITKImage image1, ITKImage image2)
{
    return perform(image1, image2, [&image1](Pixels pixels1, Pixels pixels2) {
        return divide_kernel_launch(pixels1, pixels2,
                                      image1.width,
                                      image1.height,
                                      image1.depth);
    });
}

ITKImage CudaImageOperationsProcessor::add(ITKImage image1, ITKImage image2)
{
    return perform(image1, image2, [&image1](Pixels pixels1, Pixels pixels2) {
        return add_kernel_launch(pixels1, pixels2,
                                      image1.width,
                                      image1.height,
                                      image1.depth);
    });
}

ITKImage CudaImageOperationsProcessor::perform(ITKImage image1, ITKImage image2, BinaryPixelsOperation operation)
{
    auto image1_pixels = image1.cloneToPixelArray();
    auto image2_pixels = image2.cloneToPixelArray();

    auto result_pixels = operation(image1_pixels, image2_pixels);
    auto result = ITKImage(image1.width, image1.height, image1.depth, result_pixels);

    delete[] image1_pixels;
    delete[] image2_pixels;
    delete[] result_pixels;

    return result;
}

ITKImage CudaImageOperationsProcessor::perform(ITKImage image, UnaryPixelsOperation operation)
{
    auto image_pixels = image.cloneToPixelArray();

    auto result_pixels = operation(image_pixels);

    auto result = ITKImage(image.width, image.height, image.depth, result_pixels);
    delete[] result_pixels;
    delete[] image_pixels;

    return result;
}

ITKImage CudaImageOperationsProcessor::convolution3x3(ITKImage image, ITKImage::PixelType* kernel)
{
    return perform(image, [&image, kernel](Pixels image_pixels) {
        return convolution3x3_kernel_launch(image_pixels,
                                     image.width, image.height, image.depth, kernel);
    });
}

ITKImage CudaImageOperationsProcessor::convolution3x3x3(ITKImage image, ITKImage::PixelType* kernel,
                                                        bool correct_center)
{
    return perform(image, [&image, kernel, correct_center](Pixels image_pixels) {
        return convolution3x3x3_kernel_launch(image_pixels,
                                     image.width, image.height, image.depth, kernel, correct_center);
    });
}
