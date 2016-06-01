#include "TGVLambdasProcessor.h"


template<typename Pixel>
Pixel* tgv2_l1_lambdas_launch(Pixel* f_host,
                              Pixel* lambdas_host,
                              uint width, uint height, uint depth,
                              Pixel lambda_offset,
                              Pixel lambda_factor,
                              uint iteration_count,
                              uint paint_iteration_interval,
                              TGVLambdasProcessor::IterationCallback<Pixel> iteration_finished_callback,
                              Pixel alpha0,
                              Pixel alpha1);

TGVLambdasProcessor::TGVLambdasProcessor()
{
}


ITKImage TGVLambdasProcessor::processTGV2L1LambdasGPUCuda(ITKImage input_image,
                                                          ITKImage lambdas_image,
                                                          const ITKImage::PixelType lambda_offset,
                                                          const ITKImage::PixelType lambda_factor,
                                                          const ITKImage::PixelType alpha0,
                                                          const ITKImage::PixelType alpha1,
                                                          const uint iteration_count,
                                                          const uint paint_iteration_interval, IterationFinished iteration_finished_callback)
{
    Pixel* f = input_image.cloneToPixelArray();
    Pixel* lambdas_host = lambdas_image.cloneToPixelArray();

    IterationCallback<Pixel> iteration_callback = [&input_image, iteration_finished_callback] (
            uint iteration_index, uint iteration_count, Pixel* u) {
        auto itk_u = ITKImage(input_image.width, input_image.height, input_image.depth, u);
        return iteration_finished_callback(iteration_index, iteration_count, itk_u);
    };

    Pixel* u = tgv2_l1_lambdas_launch(f, lambdas_host,
                                      input_image.width, input_image.height, input_image.depth,
                                      lambda_offset,
                                      lambda_factor,
                                      iteration_count, paint_iteration_interval,
                                      iteration_callback,
                                      alpha0,
                                      alpha1);

    delete lambdas_host;
    delete f;

    auto result = ITKImage(input_image.width, input_image.height, input_image.depth, u);
    delete u;
    return result;
}
