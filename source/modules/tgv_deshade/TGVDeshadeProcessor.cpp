#include "TGVDeshadeProcessor.h"

template<typename Pixel>
Pixel* tgv2_l1_launch(Pixel* f_host,
                      uint width, uint height, uint depth,
                      Pixel lambda,
                      uint iteration_count,
                      uint paint_iteration_interval,
                      TGVDeshadeProcessor::IterationCallback<Pixel> iteration_finished_callback,
                      Pixel alpha0,
                      Pixel alpha1);

template<typename Pixel>
Pixel* tgv2_l2_launch(Pixel* f_host,
                      uint width, uint height, uint depth,
                      Pixel lambda,
                      uint iteration_count,
                      uint paint_iteration_interval,
                      TGVDeshadeProcessor::IterationCallback<Pixel> iteration_finished_callback,
                      Pixel alpha0,
                      Pixel alpha1);

TGVDeshadeProcessor::TGVDeshadeProcessor()
{
}


ITKImage TGVDeshadeProcessor::processTVGPUCuda(ITKImage input_image,
                                        IterationFinished iteration_finished_callback,
                                        TGVAlgorithm<Pixel> tgv_algorithm)
{
    Pixel* f = input_image.cloneToPixelArray();

    IterationCallback<Pixel> iteration_callback = [&input_image, iteration_finished_callback] (
            uint iteration_index, uint iteration_count, Pixel* u) {
        auto itk_u = ITKImage(input_image.width, input_image.height, input_image.depth, u);
        return iteration_finished_callback(iteration_index, iteration_count, itk_u);
    };

    Pixel* u = tgv_algorithm(f, iteration_callback);


    delete f;

    auto result = ITKImage(input_image.width, input_image.height, input_image.depth, u);
    delete u;
    return result;
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
        (Pixel* f, IterationCallback<Pixel> iteration_callback) {
        return tgv2_l1_launch<Pixel>(f,
                                     input_image.width, input_image.height, input_image.depth,
                                     lambda,
                                     iteration_count,
                                     paint_iteration_interval,
                                     iteration_callback,
                                     alpha0, alpha1);
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
        (Pixel* f, IterationCallback<Pixel> iteration_callback) {
        return tgv2_l2_launch<Pixel>(f,
                                     input_image.width, input_image.height, input_image.depth,
                                     lambda,
                                     iteration_count,
                                     paint_iteration_interval,
                                     iteration_callback,
                                     alpha0, alpha1);
    });
}
