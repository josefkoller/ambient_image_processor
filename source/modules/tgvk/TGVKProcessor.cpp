#include "TGVKProcessor.h"

template<typename Pixel>
Pixel* tgvk_l1_launch(Pixel* f_host,
                      const uint width, const uint height, const uint depth,
                      const Pixel lambda,
                      const uint iteration_count,
                      const uint paint_iteration_interval,
                      TGVKProcessor::IterationCallback<Pixel> iteration_finished_callback,
                      const uint order,
                      const Pixel* alpha);

template<typename Pixel>
Pixel* tgv2_l1_launch(Pixel* f_host,
                  uint width, uint height, uint depth,
                  Pixel lambda,
                  uint iteration_count,
                  uint paint_iteration_interval,
                  TGVKProcessor::IterationCallback<Pixel> iteration_finished_callback,
                  Pixel alpha0,
                  Pixel alpha1);

TGVKProcessor::TGVKProcessor()
{
}


ITKImage TGVKProcessor::processTGVKL1GPUCuda(ITKImage input_image,
  const Pixel lambda,
  const uint order,
  const Pixel* alpha,
  const uint iteration_count,
  const uint paint_iteration_interval, IterationFinished iteration_finished_callback)
{
    Pixel* f = input_image.cloneToPixelArray();

    IterationCallback<Pixel> iteration_callback = [&input_image, iteration_finished_callback] (
            uint iteration_index, uint iteration_count, Pixel* u) {
        auto itk_u = ITKImage(input_image.width, input_image.height, input_image.depth, u);
        return iteration_finished_callback(iteration_index, iteration_count, itk_u);
    };

    Pixel* u = nullptr;
    if(order == 2) {
        u = tgv2_l1_launch(f,
                          input_image.width, input_image.height, input_image.depth,
                          lambda,
                          iteration_count,
                          paint_iteration_interval,
                          iteration_callback,
                          alpha[1], alpha[0]);
    } else {
        u = tgvk_l1_launch(f,
                          input_image.width, input_image.height, input_image.depth,
                          lambda,
                          iteration_count,
                          paint_iteration_interval,
                          iteration_callback,
                          order, alpha);
    }


    delete[] f;

    auto result = ITKImage(input_image.width, input_image.height, input_image.depth, u);
    delete[] u;
    return result;
}
