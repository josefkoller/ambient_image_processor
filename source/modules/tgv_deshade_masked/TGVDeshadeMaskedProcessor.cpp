#include "TGVDeshadeMaskedProcessor.h"

#include "CudaImageOperationsProcessor.h"

#include <iostream>
#include <fstream>

#include "TGVDeshadeProcessor.h"

typedef TGVDeshadeMaskedProcessor::IndexVectorConstReference IndexVectorConstReference;

template<typename Pixel>
Pixel* tgv2_l1_deshade_masked_launch(Pixel* f_host,
  uint width, uint height, uint depth,
  Pixel lambda,
  uint iteration_count,
  uint paint_iteration_interval,
  const int cuda_block_dimension,
  TGVDeshadeMaskedProcessor::IterationCallback<Pixel> iteration_finished_callback,
  Pixel alpha0,
  Pixel alpha1, Pixel** v_x_host, Pixel**v_y_host, Pixel**v_z_host,

  IndexVectorConstReference masked_pixel_indices,
  IndexVectorConstReference left_edge_pixel_indices, IndexVectorConstReference not_left_edge_pixel_indices,
  IndexVectorConstReference right_edge_pixel_indices, IndexVectorConstReference not_right_edge_pixel_indices,

  IndexVectorConstReference top_edge_pixel_indices, IndexVectorConstReference not_top_edge_pixel_indices,
  IndexVectorConstReference bottom_edge_pixel_indices, IndexVectorConstReference not_bottom_edge_pixel_indices,

  IndexVectorConstReference front_edge_pixel_indices, IndexVectorConstReference not_front_edge_pixel_indices,
  IndexVectorConstReference back_edge_pixel_indices, IndexVectorConstReference not_back_edge_pixel_indices);



TGVDeshadeMaskedProcessor::TGVDeshadeMaskedProcessor()
{
}

ITKImage TGVDeshadeMaskedProcessor::processTVGPUCuda(ITKImage input_image,
                                                     const ITKImage& mask,
                                                     const bool set_negative_values_to_zero,
                                                     const bool add_background_back,
                                                     IterationFinishedThreeImages iteration_finished_callback,
                                                     ITKImage& denoised_image,
                                                     ITKImage& shading_image,
                                                     TGVAlgorithm<Pixel> tgv_algorithm)
{
    Pixel* f = input_image.cloneToPixelArray();

    ITKImage background_mask;
    if(add_background_back && !mask.isNull())
        background_mask = CudaImageOperationsProcessor::invert(mask);

    IterationCallback<Pixel> iteration_callback =
            [add_background_back, &background_mask, &input_image, iteration_finished_callback, &mask,
            set_negative_values_to_zero] (
            uint iteration_index, uint iteration_count, Pixel* u_pixels,
            Pixel* v_x, Pixel* v_y, Pixel* v_z) {
        auto u = ITKImage(input_image.width, input_image.height, input_image.depth, u_pixels);
        auto l = ITKImage();
        auto r = TGVDeshadeProcessor::deshade_poisson_cosine_transform(u, v_x, v_y, v_z,
                                                  input_image.width, input_image.height, input_image.depth,
                                                  mask, set_negative_values_to_zero,
                                                  l);

        if(add_background_back && !mask.isNull())
        {
            auto background = CudaImageOperationsProcessor::multiply(u, background_mask);
            r = CudaImageOperationsProcessor::add(r, background);
        }

        return iteration_finished_callback(iteration_index, iteration_count, u, l, r);
    };

    Pixel* v_x, *v_y, *v_z;
    Pixel* u = tgv_algorithm(f, iteration_callback, &v_x, &v_y, &v_z);

    delete[] f;
    denoised_image = ITKImage(input_image.width, input_image.height, input_image.depth, u);
    delete[] u;
    auto deshaded_image = TGVDeshadeProcessor::deshade_poisson_cosine_transform(denoised_image, v_x, v_y, v_z,
                                                           input_image.width, input_image.height, input_image.depth,
                                                           mask, set_negative_values_to_zero,
                                                           shading_image, true);

    delete[] v_x;
    delete[] v_y;
    if(input_image.depth > 1)
        delete[] v_z;

    if(add_background_back && !mask.isNull())
    {
        auto background = CudaImageOperationsProcessor::multiply(denoised_image, background_mask);
        deshaded_image = CudaImageOperationsProcessor::add(deshaded_image, background);
    }

    return deshaded_image;
}


ITKImage TGVDeshadeMaskedProcessor::processTGV2L1GPUCuda(ITKImage input_image,
                                                         const Pixel lambda,
                                                         const Pixel alpha0,
                                                         const Pixel alpha1,
                                                         const uint iteration_count,
                                                         const int cuda_block_dimension,
                                                         const uint paint_iteration_interval,
                                                         IterationFinishedThreeImages iteration_finished_callback,
                                                         const ITKImage& mask,
                                                         const bool set_negative_values_to_zero,
                                                         const bool add_background_back,
                                                         ITKImage& denoised_image,
                                                         ITKImage& shading_image)
{
    return processTVGPUCuda(input_image, mask, set_negative_values_to_zero, add_background_back, iteration_finished_callback,
                            denoised_image, shading_image,
                            [&input_image, lambda, iteration_count, paint_iteration_interval, alpha0, alpha1,
                            add_background_back, &mask, cuda_block_dimension]
                            (Pixel* f, IterationCallback<Pixel> iteration_callback,
                            Pixel** v_x, Pixel**v_y, Pixel**v_z) {

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

        return tgv2_l1_deshade_masked_launch<Pixel>(f,
                                                    input_image.width, input_image.height, input_image.depth,
                                                    lambda,
                                                    iteration_count,
                                                    paint_iteration_interval,
                                                    cuda_block_dimension,
                                                    iteration_callback,
                                                    alpha0, alpha1,
                                                    v_x, v_y, v_z,

                                                    pixel_indices_inside_mask,
                                                    left_edge_pixel_indices, not_left_edge_pixel_indices,
                                                    right_edge_pixel_indices, not_right_edge_pixel_indices,

                                                    top_edge_pixel_indices, not_top_edge_pixel_indices,
                                                    bottom_edge_pixel_indices, not_bottom_edge_pixel_indices,

                                                    front_edge_pixel_indices, not_front_edge_pixel_indices,
                                                    back_edge_pixel_indices, not_back_edge_pixel_indices);
    });
}

void TGVDeshadeMaskedProcessor::buildMaskIndices(ITKImage mask, 
                                                 IndexVector& pixel_indices_inside_mask,
                                                 IndexVector& left_edge_pixel_indices, IndexVector& not_left_edge_pixel_indices,
                                                 IndexVector& right_edge_pixel_indices, IndexVector& not_right_edge_pixel_indices,

                                                 IndexVector& top_edge_pixel_indices, IndexVector& not_top_edge_pixel_indices,
                                                 IndexVector& bottom_edge_pixel_indices, IndexVector& not_bottom_edge_pixel_indices,

                                                 IndexVector& front_edge_pixel_indices, IndexVector& not_front_edge_pixel_indices,
                                                 IndexVector& back_edge_pixel_indices, IndexVector& not_back_edge_pixel_indices)
{
    mask.foreachPixel([&](uint x, uint y, uint z, ITKImage::PixelType mask_value) {
        if(mask_value < 1)
            return;
        uint linear_index = mask.linearIndex(x, y, z);

        pixel_indices_inside_mask.push_back(linear_index);

        if(x == 0)
            left_edge_pixel_indices.push_back(linear_index);
        else
            not_left_edge_pixel_indices.push_back(linear_index);

        if(x == mask.width - 1)
            right_edge_pixel_indices.push_back(linear_index);
        else
            not_right_edge_pixel_indices.push_back(linear_index);

        if(y == 0)
            top_edge_pixel_indices.push_back(linear_index);
        else
            not_top_edge_pixel_indices.push_back(linear_index);

        if(y == mask.height - 1)
            bottom_edge_pixel_indices.push_back(linear_index);
        else
            not_bottom_edge_pixel_indices.push_back(linear_index);

        if(z == 0)
            front_edge_pixel_indices.push_back(linear_index);
        else
            not_front_edge_pixel_indices.push_back(linear_index);

        if(z == mask.depth - 1)
            back_edge_pixel_indices.push_back(linear_index);
        else
            not_back_edge_pixel_indices.push_back(linear_index);
    });
}
