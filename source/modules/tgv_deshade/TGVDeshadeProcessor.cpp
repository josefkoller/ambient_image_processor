#include "TGVDeshadeProcessor.h"

template<typename Pixel>
Pixel* tgv2_l1_deshade_launch(Pixel* f_host,
                      uint width, uint height, uint depth,
                      Pixel lambda,
                      uint iteration_count,
                      uint paint_iteration_interval,
                      TGVDeshadeProcessor::IterationCallback<Pixel> iteration_finished_callback,
                      Pixel alpha0,
                      Pixel alpha1, Pixel** v_x, Pixel**v_y, Pixel**v_z);

template<typename Pixel>
Pixel* tgv2_l2_deshade_launch(Pixel* f_host,
                      uint width, uint height, uint depth,
                      Pixel lambda,
                      uint iteration_count,
                      uint paint_iteration_interval,
                      TGVDeshadeProcessor::IterationCallback<Pixel> iteration_finished_callback,
                      Pixel alpha0,
                      Pixel alpha1, Pixel** v_x, Pixel**v_y, Pixel**v_z);

TGVDeshadeProcessor::TGVDeshadeProcessor()
{
}


ITKImage TGVDeshadeProcessor::processTVGPUCuda(ITKImage input_image,
                                        IterationFinished iteration_finished_callback,
                                        TGVAlgorithm<Pixel> tgv_algorithm)
{
    Pixel* f = input_image.cloneToPixelArray();

    IterationCallback<Pixel> iteration_callback = [&input_image, iteration_finished_callback] (
            uint iteration_index, uint iteration_count, Pixel* u,
            Pixel* v_x, Pixel* v_y, Pixel* v_z) {

        auto itk_u = ITKImage(input_image.width, input_image.height, input_image.depth, u);
        auto itk_v_x = ITKImage(input_image.width, input_image.height, input_image.depth, v_x);
        auto itk_v_y = ITKImage(input_image.width, input_image.height, input_image.depth, v_y);
        auto itk_v_z = input_image.depth == 1 ? itk_u.cloneSameSizeWithZeros() :
                       ITKImage(input_image.width, input_image.height, input_image.depth, v_z);
        auto l = integrate_image_gradients(itk_v_x, itk_v_y, itk_v_z);

        return iteration_finished_callback(iteration_index, iteration_count, l);
    };

    Pixel* v_x, *v_y, *v_z;
    Pixel* u = tgv_algorithm(f, iteration_callback, &v_x, &v_y, &v_z);


    delete f;

    auto itk_u = ITKImage(input_image.width, input_image.height, input_image.depth, u);
    auto itk_v_x = ITKImage(input_image.width, input_image.height, input_image.depth, v_x);
    auto itk_v_y = ITKImage(input_image.width, input_image.height, input_image.depth, v_y);
    auto itk_v_z = input_image.depth == 1 ? itk_u.cloneSameSizeWithZeros() :
                   ITKImage(input_image.width, input_image.height, input_image.depth, v_z);
    auto l = integrate_image_gradients(itk_v_x, itk_v_y, itk_v_z);

    delete v_x;
    delete v_y;
    if(input_image.depth > 1)
        delete v_z;
    delete u; // TODO: u is not used by now
    return l;
}


ITKImage TGVDeshadeProcessor::processTGV2L1GPUCuda(ITKImage input_image,
                                          const Pixel lambda,
                                          const Pixel alpha0,
                                          const Pixel alpha1,
                                          const uint iteration_count,
                                          const uint paint_iteration_interval,
                                          IterationFinished iteration_finished_callback)
{
    return processTVGPUCuda(input_image, iteration_finished_callback,
        [&input_image, lambda, iteration_count, paint_iteration_interval, alpha0, alpha1]
        (Pixel* f, IterationCallback<Pixel> iteration_callback, Pixel** v_x, Pixel**v_y, Pixel**v_z) {
        return tgv2_l1_deshade_launch<Pixel>(f,
                                     input_image.width, input_image.height, input_image.depth,
                                     lambda,
                                     iteration_count,
                                     paint_iteration_interval,
                                     iteration_callback,
                                     alpha0, alpha1,
                                     v_x, v_y, v_z);
    });
}

ITKImage TGVDeshadeProcessor::processTGV2L2GPUCuda(ITKImage input_image,
                                          const Pixel lambda,
                                          const Pixel alpha0,
                                          const Pixel alpha1,
                                          const uint iteration_count,
                                          const uint paint_iteration_interval,
                                          IterationFinished iteration_finished_callback)
{
    return processTVGPUCuda(input_image, iteration_finished_callback,
        [&input_image, lambda, iteration_count, paint_iteration_interval, alpha0, alpha1]
        (Pixel* f, IterationCallback<Pixel> iteration_callback, Pixel** v_x, Pixel**v_y, Pixel**v_z) {
        return tgv2_l1_deshade_launch<Pixel>(f, // TODO L2
                                    input_image.width, input_image.height, input_image.depth,
                                     lambda,
                                     iteration_count,
                                     paint_iteration_interval,
                                     iteration_callback,
                                     alpha0, alpha1,
                                     v_x, v_y, v_z);
    });
}

ITKImage TGVDeshadeProcessor::integrate_image_gradients(ITKImage gradient_x, ITKImage gradient_y, ITKImage gradient_z)
{
    ITKImage image = gradient_x.cloneSameSizeWithZeros();

    for(uint z = 0; z < image.depth; z++)
    {
        for(uint x = 1; x < image.width; x++)
        {
            for(uint y = 0; y < image.height; y++)
            {
                ITKImage::PixelType value = image.getPixel(x-1, y, z) + gradient_x.getPixel(x,y,z);
                image.setPixel(x,y,z, value);
            }
        }
    }

    for(uint z = 0; z < image.depth; z++)
    {
        for(uint x = 0; x < image.width; x++)
        {
            for(uint y = 1; y < image.height; y++)
            {
                ITKImage::PixelType value = image.getPixel(x, y-1, z) + gradient_y.getPixel(x,y,z);
                image.setPixel(x,y,z, value);
            }
        }
    }

    if(image.depth == 1)
        return image;

    for(uint z = 1; z < image.depth; z++)
    {
        for(uint x = 0; x < image.width; x++)
        {
            for(uint y = 0; y < image.height; y++)
            {
                ITKImage::PixelType value = image.getPixel(x, y, z-1) + gradient_z.getPixel(x,y,z);
                image.setPixel(x,y,z, value);
            }
        }
    }

    return image;
}
