#include "TGVDeshadeProcessor.h"

#include "CudaImageOperationsProcessor.h"

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

ITKImage TGVDeshadeProcessor::deshade(Pixel* u, Pixel* v_x, Pixel* v_y, Pixel* v_z,
                                      const uint width,
                                      const uint height,
                                      const uint depth)
{
    auto itk_u = ITKImage(width, height, depth, u);
    auto itk_v_x = ITKImage(width, height, depth, v_x);
    auto itk_v_y = ITKImage(width, height, depth, v_y);
    auto itk_v_z = depth == 1 ? itk_u.cloneSameSizeWithZeros() :
                   ITKImage(width, height, depth, v_z);
    auto l = integrate_image_gradients(itk_v_x, itk_v_y, itk_v_z);

    return CudaImageOperationsProcessor::subtract(itk_u, l);
}

ITKImage TGVDeshadeProcessor::deshade_poisson_cosine_transform(Pixel* u, Pixel* v_x, Pixel* v_y, Pixel* v_z,
                                       const uint width,
                                       const uint height,
                                       const uint depth, bool is_host_data)
{
    auto itk_u = ITKImage(width, height, depth, u);


    /*
    return  CudaImageOperationsProcessor::inverseCosineTransform(
                CudaImageOperationsProcessor::cosineTransform(itk_u));
    */




    auto l = integrate_image_gradients_poisson_cosine_transform(v_x, v_y, v_z,
                                                                width, height, depth, is_host_data);

    return l;

    return CudaImageOperationsProcessor::subtract(itk_u, l);
}

ITKImage TGVDeshadeProcessor::processTVGPUCuda(ITKImage input_image,
                                        IterationFinished iteration_finished_callback,
                                        TGVAlgorithm<Pixel> tgv_algorithm)
{
    Pixel* f = input_image.cloneToPixelArray();

    IterationCallback<Pixel> iteration_callback = [&input_image, iteration_finished_callback] (
            uint iteration_index, uint iteration_count, Pixel* u,
            Pixel* v_x, Pixel* v_y, Pixel* v_z) {
        auto r = deshade_poisson_cosine_transform(u, v_x, v_y, v_z,
                                                  input_image.width, input_image.height, input_image.depth);
        return iteration_finished_callback(iteration_index, iteration_count, r);
    };

    Pixel* v_x, *v_y, *v_z;
    Pixel* u = tgv_algorithm(f, iteration_callback, &v_x, &v_y, &v_z);


    delete[] f;

    auto r = deshade_poisson_cosine_transform(u, v_x, v_y, v_z,
                                              input_image.width, input_image.height, input_image.depth, true);

    delete[] v_x;
    delete[] v_y;
    if(input_image.depth > 1)
        delete[] v_z;
    delete[] u;

    return r;
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

ITKImage TGVDeshadeProcessor::integrate_image_gradients_poisson_cosine_transform(Pixel* gradient_x,
                                                                                 Pixel* gradient_y,
                                                                                 Pixel* gradient_z,
                                                                                 const uint width,
                                                                                 const uint height,
                                                                                 const uint depth,
                                                                                 bool is_host_data)
{
    Pixel* divergence = CudaImageOperationsProcessor::divergence(gradient_x, gradient_y, gradient_z,
                                                                 width, height, depth, is_host_data);
    ITKImage divergence_image = ITKImage(width, height, depth, divergence);
    delete[] divergence;

 // return divergence_image;

    ITKImage divergence_image_cosine = CudaImageOperationsProcessor::cosineTransform(divergence_image);

 // return divergence_image_cosine;

    ITKImage result_cosine_domain = CudaImageOperationsProcessor::solvePoissonInCosineDomain(divergence_image_cosine);

    return CudaImageOperationsProcessor::inverseCosineTransform(result_cosine_domain); // inverse
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
