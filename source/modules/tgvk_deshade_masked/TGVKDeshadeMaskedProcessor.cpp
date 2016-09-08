#include "TGVKDeshadeMaskedProcessor.h"

#include "CudaImageOperationsProcessor.h"

#include "TGVDeshadeMaskedProcessor.h"
#include "TGVDeshadeProcessor.h"

typedef TGVKDeshadeMaskedProcessor::IndexVectorConstReference IndexVectorConstReference;

template<typename Pixel>
Pixel* tgvk_l1_deshade_masked_launch(Pixel* f_host,
  const uint width, const uint height, const uint depth,
  const Pixel lambda,
  const uint iteration_count,
  const uint paint_iteration_interval,
  const int cuda_block_dimension,
  TGVKDeshadeMaskedProcessor::IterationCallback<Pixel> iteration_finished_callback,
  const uint order,
  const Pixel* alpha,
  Pixel** v_x_host, Pixel**v_y_host, Pixel**v_z_host,

  IndexVectorConstReference masked_pixel_indices,
  IndexVectorConstReference left_edge_pixel_indices, IndexVectorConstReference not_left_edge_pixel_indices,
  IndexVectorConstReference right_edge_pixel_indices, IndexVectorConstReference not_right_edge_pixel_indices,

  IndexVectorConstReference top_edge_pixel_indices, IndexVectorConstReference not_top_edge_pixel_indices,
  IndexVectorConstReference bottom_edge_pixel_indices, IndexVectorConstReference not_bottom_edge_pixel_indices,

  IndexVectorConstReference front_edge_pixel_indices, IndexVectorConstReference not_front_edge_pixel_indices,
  IndexVectorConstReference back_edge_pixel_indices, IndexVectorConstReference not_back_edge_pixel_indices);


template<typename Pixel>
Pixel* tgvk_l1_deshade_masked_2d_launch(Pixel* f_host,
  const uint width, const uint height,
  const Pixel lambda,
  const uint iteration_count,
  const uint paint_iteration_interval,
  const int cuda_block_dimension,
  TGVDeshadeMaskedProcessor::IterationCallback2D<Pixel> iteration_finished_callback,
  const uint order,
  const Pixel* alpha,
  Pixel** v_x_host, Pixel**v_y_host,

  IndexVectorConstReference masked_pixel_indices,
  IndexVectorConstReference left_edge_pixel_indices, IndexVectorConstReference not_left_edge_pixel_indices,
  IndexVectorConstReference right_edge_pixel_indices, IndexVectorConstReference not_right_edge_pixel_indices,

  IndexVectorConstReference top_edge_pixel_indices, IndexVectorConstReference not_top_edge_pixel_indices,
  IndexVectorConstReference bottom_edge_pixel_indices, IndexVectorConstReference not_bottom_edge_pixel_indices);

TGVKDeshadeMaskedProcessor::TGVKDeshadeMaskedProcessor()
{
}

void TGVKDeshadeMaskedProcessor::processTGVKL1Cuda(ITKImage input_image,
                             const Pixel lambda,

                             const uint order,
                             const Pixel* alpha,

                             const uint iteration_count,
                             const int cuda_block_dimension,
                             ITKImage mask,
                             const bool set_negative_values_to_zero,
                             const bool add_background_back,

                             const uint paint_iteration_interval,
                             IterationFinishedThreeImages iteration_finished_callback,

                             ITKImage& denoised_image,
                             ITKImage& shading_image,
                             ITKImage& deshaded_image,
                             ITKImage& div_v_image,
                             const bool calculate_div_v)
{
    if(order == 2) {
        deshaded_image = TGVDeshadeMaskedProcessor::processTGV2L1GPUCuda(input_image, lambda, alpha[1], alpha[0],
                iteration_count, cuda_block_dimension, paint_iteration_interval, iteration_finished_callback,
                mask, set_negative_values_to_zero, add_background_back, denoised_image, shading_image,
                div_v_image, calculate_div_v);
        return;
    }

    if(input_image.depth == 1) {
        processTGVKL1Cuda2D(input_image, lambda, order, alpha, iteration_count, cuda_block_dimension,
                          mask, set_negative_values_to_zero, add_background_back, paint_iteration_interval,
                          iteration_finished_callback, denoised_image, shading_image, deshaded_image,
                          div_v_image, calculate_div_v);
        return;
    }


    Pixel* f = input_image.cloneToPixelArray();

    if(mask.isNull())
    {
        mask = input_image.cloneSameSizeWithZeros();
        mask.setEachPixel([](uint,uint,uint) { return 1.0; });
    }

    ITKImage background_mask;
    if(add_background_back && !mask.isNull())
      background_mask = CudaImageOperationsProcessor::invert(mask);

    IterationCallback<Pixel> iteration_callback = nullptr;

    if(iteration_finished_callback != nullptr)
        iteration_callback =
        [add_background_back, &background_mask, &input_image, iteration_finished_callback, &mask,
        set_negative_values_to_zero] (
        uint iteration_index, uint iteration_count, Pixel* u_pixels,
        Pixel* v_x, Pixel* v_y, Pixel* v_z) {
            auto u = ITKImage(input_image.width, input_image.height, input_image.depth, u_pixels);
            auto l = ITKImage();
            auto r = TGVDeshadeProcessor::deshade_poisson_cosine_transform(u, v_x, v_y, v_z,
                                                      input_image.width, input_image.height, input_image.depth,
                                                      mask, set_negative_values_to_zero,
                                                      l, true);

            if(add_background_back && !mask.isNull())
            {
                auto background = CudaImageOperationsProcessor::multiply(u, background_mask);
                r = CudaImageOperationsProcessor::add(r, background);
            }

            return iteration_finished_callback(iteration_index, iteration_count, u, l, r);
    };

    Pixel* v_x, *v_y, *v_z;
    Pixel* u = nullptr;

    IndexVector left_edge_pixel_indices, not_left_edge_pixel_indices,
            right_edge_pixel_indices, not_right_edge_pixel_indices,
            top_edge_pixel_indices, not_top_edge_pixel_indices,
            bottom_edge_pixel_indices, not_bottom_edge_pixel_indices,
            front_edge_pixel_indices, not_front_edge_pixel_indices,
            back_edge_pixel_indices, not_back_edge_pixel_indices,
            pixel_indices_inside_mask;
     TGVDeshadeMaskedProcessor::buildMaskIndices(mask,
                     pixel_indices_inside_mask,
                     left_edge_pixel_indices, not_left_edge_pixel_indices,
                     right_edge_pixel_indices, not_right_edge_pixel_indices,

                     top_edge_pixel_indices, not_top_edge_pixel_indices,
                     bottom_edge_pixel_indices, not_bottom_edge_pixel_indices,

                     front_edge_pixel_indices, not_front_edge_pixel_indices,
                     back_edge_pixel_indices, not_back_edge_pixel_indices);

    u = tgvk_l1_deshade_masked_launch(f,
                              input_image.width, input_image.height, input_image.depth,
                              lambda,
                              iteration_count,
                              paint_iteration_interval,
                              cuda_block_dimension,
                              iteration_callback,
                              order, alpha,
                              &v_x, &v_y, &v_z,

                              pixel_indices_inside_mask,
                              left_edge_pixel_indices, not_left_edge_pixel_indices,
                              right_edge_pixel_indices, not_right_edge_pixel_indices,

                              top_edge_pixel_indices, not_top_edge_pixel_indices,
                              bottom_edge_pixel_indices, not_bottom_edge_pixel_indices,

                              front_edge_pixel_indices, not_front_edge_pixel_indices,
                              back_edge_pixel_indices, not_back_edge_pixel_indices);

    delete[] f;
    denoised_image = ITKImage(input_image.width, input_image.height, input_image.depth, u);
    delete[] u;
    deshaded_image = TGVDeshadeProcessor::deshade_poisson_cosine_transform(denoised_image, v_x, v_y, v_z,
                                                           input_image.width, input_image.height, input_image.depth,
                                                           mask, set_negative_values_to_zero,
                                                           shading_image, true);

    if(calculate_div_v)
    {
        Pixel* divergence = CudaImageOperationsProcessor::divergence(v_x, v_y, v_z,
                                                                     input_image.width, input_image.height, input_image.depth,
                                                                     true);
        div_v_image = ITKImage(input_image.width, input_image.height, input_image.depth, divergence);
        delete[] divergence;
    }

    delete[] v_x;
    delete[] v_y;
    delete[] v_z;

    if(add_background_back && !mask.isNull())
    {
        auto background = CudaImageOperationsProcessor::multiply(denoised_image, background_mask);
        deshaded_image = CudaImageOperationsProcessor::add(deshaded_image, background);
    }
}


void TGVKDeshadeMaskedProcessor::processTGVKL1Cuda2D(ITKImage input_image,
                             const Pixel lambda,

                             const uint order,
                             const Pixel* alpha,

                             const uint iteration_count,
                             const int cuda_block_dimension,
                             ITKImage mask,
                             const bool set_negative_values_to_zero,
                             const bool add_background_back,

                             const uint paint_iteration_interval,
                             IterationFinishedThreeImages iteration_finished_callback,

                             ITKImage& denoised_image,
                             ITKImage& shading_image,
                             ITKImage& deshaded_image,
                             ITKImage& div_v_image,
                             const bool calculate_div_v)
{
    if(order == 2) {
        deshaded_image = TGVDeshadeMaskedProcessor::processTGV2L1GPUCuda2D(input_image, lambda, alpha[0], alpha[1],
                iteration_count, cuda_block_dimension, paint_iteration_interval, iteration_finished_callback,
                mask, set_negative_values_to_zero, add_background_back, denoised_image, shading_image,
                div_v_image, calculate_div_v);
        return;
    }

    Pixel* f = input_image.cloneToPixelArray();

    if(mask.isNull())
    {
        mask = input_image.cloneSameSizeWithZeros();
        mask.setEachPixel([](uint,uint,uint) { return 1.0; });
    }

    ITKImage background_mask;
    if(add_background_back && !mask.isNull())
      background_mask = CudaImageOperationsProcessor::invert(mask);

    IterationCallback2D<Pixel> iteration_callback = nullptr;

    if(iteration_finished_callback != nullptr)
        iteration_callback =
        [add_background_back, &background_mask, &input_image, iteration_finished_callback, &mask,
        set_negative_values_to_zero] (
        uint iteration_index, uint iteration_count, Pixel* u_pixels,
        Pixel* v_x, Pixel* v_y) {
            auto u = ITKImage(input_image.width, input_image.height, input_image.depth, u_pixels);
            auto l = ITKImage();
            auto r = TGVDeshadeProcessor::deshade_poisson_cosine_transform_2d(u, v_x, v_y,
                                                      input_image.width, input_image.height,
                                                      mask, set_negative_values_to_zero,
                                                      l, true);

            if(add_background_back && !mask.isNull())
            {
                auto background = CudaImageOperationsProcessor::multiply(u, background_mask);
                r = CudaImageOperationsProcessor::add(r, background);
            }

            return iteration_finished_callback(iteration_index, iteration_count, u, l, r);
    };

    Pixel* v_x, *v_y;
    Pixel* u = nullptr;

    IndexVector left_edge_pixel_indices, not_left_edge_pixel_indices,
            right_edge_pixel_indices, not_right_edge_pixel_indices,
            top_edge_pixel_indices, not_top_edge_pixel_indices,
            bottom_edge_pixel_indices, not_bottom_edge_pixel_indices,
            pixel_indices_inside_mask;
     TGVDeshadeMaskedProcessor::buildMaskIndices2D(mask,
                     pixel_indices_inside_mask,
                     left_edge_pixel_indices, not_left_edge_pixel_indices,
                     right_edge_pixel_indices, not_right_edge_pixel_indices,

                     top_edge_pixel_indices, not_top_edge_pixel_indices,
                     bottom_edge_pixel_indices, not_bottom_edge_pixel_indices);

    u = tgvk_l1_deshade_masked_2d_launch(f,
                              input_image.width, input_image.height,
                              lambda,
                              iteration_count,
                              paint_iteration_interval,
                              cuda_block_dimension,
                              iteration_callback,
                              order, alpha,
                              &v_x, &v_y,

                              pixel_indices_inside_mask,
                              left_edge_pixel_indices, not_left_edge_pixel_indices,
                              right_edge_pixel_indices, not_right_edge_pixel_indices,

                              top_edge_pixel_indices, not_top_edge_pixel_indices,
                              bottom_edge_pixel_indices, not_bottom_edge_pixel_indices);

    delete[] f;
    denoised_image = ITKImage(input_image.width, input_image.height, input_image.depth, u);
    delete[] u;
    deshaded_image = TGVDeshadeProcessor::deshade_poisson_cosine_transform_2d(denoised_image, v_x, v_y,
                                                           input_image.width, input_image.height,
                                                           mask, set_negative_values_to_zero,
                                                           shading_image, true);

    if(calculate_div_v)
    {
        Pixel* divergence = CudaImageOperationsProcessor::divergence_2d(v_x, v_y,
                                                                     input_image.width, input_image.height,
                                                                     true);
        div_v_image = ITKImage(input_image.width, input_image.height, input_image.depth, divergence);
        delete[] divergence;
    }

    delete[] v_x;
    delete[] v_y;

    if(add_background_back && !mask.isNull())
    {
        auto background = CudaImageOperationsProcessor::multiply(denoised_image, background_mask);
        deshaded_image = CudaImageOperationsProcessor::add(deshaded_image, background);
    }
}
