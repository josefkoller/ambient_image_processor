#include "TGVDeshadeProcessor.h"

#include "CudaImageOperationsProcessor.h"

#include <iostream>
#include <fstream>

template<typename Pixel>
Pixel* tgv2_l1_deshade_launch(Pixel* f_host,
                              uint width, uint height, uint depth,
                              Pixel lambda,
                              uint iteration_count,
                              uint paint_iteration_interval,
                              const int cuda_block_dimension,
                              TGVDeshadeProcessor::IterationCallback<Pixel> iteration_finished_callback,
                              Pixel alpha0,
                              Pixel alpha1,
                              Pixel** v_x, Pixel**v_y, Pixel**v_z);

template<typename Pixel>
Pixel* tgv2_l1_deshade_launch_2d(Pixel* f_host,
                              uint width, uint height,
                              Pixel lambda,
                              uint iteration_count,
                              uint paint_iteration_interval,
                              TGVDeshadeProcessor::IterationCallback2D<Pixel> iteration_finished_callback,
                              Pixel alpha0,
                              Pixel alpha1,
                              Pixel** v_x, Pixel**v_y);

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

ITKImage TGVDeshadeProcessor::deshade_poisson_cosine_transform(ITKImage u, Pixel* v_x, Pixel* v_y, Pixel* v_z,
                                                               const uint width,
                                                               const uint height,
                                                               const uint depth,
                                                               const ITKImage& mask,
                                                               const bool set_negative_values_to_zero,
                                                               ITKImage& l,
                                                               bool is_host_data)
{
    l = integrate_image_gradients_poisson_cosine_transform(v_x, v_y, v_z,
                                                           width, height, depth,
                                                           is_host_data);
    auto r = CudaImageOperationsProcessor::subtract(u, l);

    if(!mask.isNull())
    {
        l = CudaImageOperationsProcessor::multiply(l, mask);
        r = CudaImageOperationsProcessor::multiply(r, mask);
    }

    if(set_negative_values_to_zero)
        r = CudaImageOperationsProcessor::clamp_negative_values(r, 0);

    return r;
}

ITKImage TGVDeshadeProcessor::deshade_poisson_cosine_transform_2d(ITKImage u, Pixel* v_x, Pixel* v_y,
                                                               const uint width,
                                                               const uint height,
                                                               const ITKImage& mask,
                                                               const bool set_negative_values_to_zero,
                                                               ITKImage& l,
                                                               bool is_host_data)
{
    l = integrate_image_gradients_poisson_cosine_transform_2d(v_x, v_y,
                                                           width, height,
                                                           is_host_data);
    auto r = CudaImageOperationsProcessor::subtract(u, l);

    if(!mask.isNull())
    {
        l = CudaImageOperationsProcessor::multiply(l, mask);
        r = CudaImageOperationsProcessor::multiply(r, mask);
    }

    if(set_negative_values_to_zero)
        r = CudaImageOperationsProcessor::clamp_negative_values(r, 0);

    return r;
}

void TGVDeshadeProcessor::processTGV2L1GPUCuda(ITKImage input_image,
                                               const Pixel lambda,
                                               const Pixel alpha0,
                                               const Pixel alpha1,
                                               const uint iteration_count,
                                               const int cuda_block_dimension,
                                               const ITKImage& mask_image,
                                               const bool set_negative_values_to_zero,
                                               ITKImage& denoised_image,
                                               ITKImage& shading_image,
                                               ITKImage& deshaded_image)
{
    Pixel* f = input_image.cloneToPixelArray();

    Pixel* v_x, *v_y, *v_z;
    Pixel* u = tgv2_l1_deshade_launch<Pixel>(f,
                                             input_image.width, input_image.height, input_image.depth,
                                             lambda,
                                             iteration_count,
                                             iteration_count + 1,
                                             cuda_block_dimension,
                                             nullptr,
                                             alpha0, alpha1,
                                             &v_x, &v_y, &v_z);
    delete[] f;

    denoised_image = ITKImage(input_image.width, input_image.height, input_image.depth, u);
    delete[] u;

    shading_image = integrate_image_gradients_poisson_cosine_transform(v_x, v_y, v_z,
                                                                       input_image.width, input_image.height, input_image.depth,
                                                                       true);
    delete[] v_x;
    delete[] v_y;
    if(input_image.depth > 1)
        delete[] v_z;

    deshaded_image = CudaImageOperationsProcessor::subtract(denoised_image, shading_image);

    if(!mask_image.isNull())
        deshaded_image = CudaImageOperationsProcessor::multiply(deshaded_image, mask_image);

    if(set_negative_values_to_zero)
        deshaded_image = CudaImageOperationsProcessor::clamp_negative_values(deshaded_image, 0);
}

ITKImage::PixelType TGVDeshadeProcessor::mean(const ITKImage& image)
{
    ITKImage::PixelType mean = 0;
    image.foreachPixel([&mean](uint, uint, uint, ITKImage::PixelType pixel) {
        mean+= pixel;
    });
    return mean /= image.voxel_count;
}

ITKImage::PixelType TGVDeshadeProcessor::standard_deviation(const ITKImage& image, const ITKImage::PixelType mean)
{
    ITKImage::PixelType standard_deviation = 0;
    image.foreachPixel([mean, &standard_deviation](uint, uint, uint, ITKImage::PixelType pixel) {
        const auto difference = pixel - mean;
        standard_deviation+= difference * difference;
    });
    return std::sqrt(standard_deviation / (image.voxel_count - 1));
}

ITKImage::PixelType TGVDeshadeProcessor::normalized_cross_correlation(const ITKImage& image1,
                                                                      const ITKImage& image2)
{
    const auto mean1 = mean(image1);
    const auto std1 = standard_deviation(image1, mean1);
    const auto mean2 = mean(image1);
    const auto std2 = standard_deviation(image2, mean2);

    /*
    printf("mean1: %f \n", mean1);
    printf("std1: %f \n", std1);
    printf("mean2: %f \n", mean2);
    printf("std2: %f \n", std2);
    */

    ITKImage::PixelType normalized_cross_correlation = 0;
    image1.foreachPixel([mean1, mean2, &image2, &normalized_cross_correlation] (uint x, uint y, uint z, ITKImage::PixelType pixel1) {
        const auto difference1 = pixel1 - mean1;
        const auto pixel2 = image2.getPixel(x,y,z);
        const auto difference2 = pixel2 - mean2;
        normalized_cross_correlation+= difference1 * difference2;
    });
    return normalized_cross_correlation / (std1 * std2 * image1.voxel_count);
}

ITKImage::PixelType TGVDeshadeProcessor::time_tv(ITKImage image, ITKImage image_before)
{
    auto image_change = CudaImageOperationsProcessor::subtract(image, image_before);
    ITKImage::PixelType abs_change_sum = 0;
    image_change.foreachPixel([&abs_change_sum](uint, uint, uint, ITKImage::PixelType v) {
        abs_change_sum += std::abs(v);
    });
    return abs_change_sum;
}



ITKImage TGVDeshadeProcessor::processTVGPUCuda(ITKImage input_image,
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
        auto r = deshade_poisson_cosine_transform(u, v_x, v_y, v_z,
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
    Pixel* u = tgv_algorithm(f, iteration_callback, &v_x, &v_y, &v_z);

    delete[] f;
    denoised_image = ITKImage(input_image.width, input_image.height, input_image.depth, u);
    delete[] u;
    auto deshaded_image = deshade_poisson_cosine_transform(denoised_image, v_x, v_y, v_z,
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


ITKImage TGVDeshadeProcessor::processTGV2L1GPUCuda(ITKImage input_image,
                                                   const Pixel lambda,
                                                   const Pixel alpha0,
                                                   const Pixel alpha1,
                                                   const uint iteration_count,
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
                            add_background_back]
                            (Pixel* f, IterationCallback<Pixel> iteration_callback,
                            Pixel** v_x, Pixel**v_y, Pixel**v_z) {
        return tgv2_l1_deshade_launch<Pixel>(f,
                                             input_image.width, input_image.height, input_image.depth,
                                             lambda,
                                             iteration_count,
                                             paint_iteration_interval,
                                             -1,
                                             iteration_callback,
                                             alpha0, alpha1,
                                             v_x, v_y, v_z);
    });
}

ITKImage TGVDeshadeProcessor::processTGV2L1GPUCuda2D(ITKImage input_image,
                                                   const Pixel lambda,
                                                   const Pixel alpha0,
                                                   const Pixel alpha1,
                                                   const uint iteration_count,
                                                   const uint paint_iteration_interval,
                                                   IterationFinishedThreeImages iteration_finished_callback,
                                                   const ITKImage& mask,
                                                   const bool set_negative_values_to_zero,
                                                   const bool add_background_back,
                                                   ITKImage& denoised_image,
                                                   ITKImage& shading_image)
{
    Pixel* f = input_image.cloneToPixelArray();

    ITKImage background_mask;
    if(add_background_back && !mask.isNull())
      background_mask = CudaImageOperationsProcessor::invert(mask);

    IterationCallback2D<Pixel> iteration_callback =
            [add_background_back, &background_mask, &input_image, iteration_finished_callback, &mask,
            set_negative_values_to_zero] (
            uint iteration_index, uint iteration_count, Pixel* u_pixels,
            Pixel* v_x, Pixel* v_y) {
        auto u = ITKImage(input_image.width, input_image.height, input_image.depth, u_pixels);
        auto l = ITKImage();
        auto r = deshade_poisson_cosine_transform_2d(u, v_x, v_y,
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

    Pixel* u = tgv2_l1_deshade_launch_2d<Pixel>(f,
                                             input_image.width, input_image.height,
                                             lambda,
                                             iteration_count,
                                             paint_iteration_interval,
                                             iteration_callback,
                                             alpha0, alpha1,
                                             &v_x, &v_y);

    delete[] f;
    denoised_image = ITKImage(input_image.width, input_image.height, input_image.depth, u);
    delete[] u;
    auto deshaded_image = deshade_poisson_cosine_transform_2d(denoised_image, v_x, v_y,
                                                           input_image.width, input_image.height,
                                                           mask, set_negative_values_to_zero,
                                                           shading_image, true);
    delete[] v_x;
    delete[] v_y;

    if(add_background_back && !mask.isNull())
    {
        auto background = CudaImageOperationsProcessor::multiply(denoised_image, background_mask);
        deshaded_image = CudaImageOperationsProcessor::add(deshaded_image, background);
    }

    return deshaded_image;
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

ITKImage TGVDeshadeProcessor::integrate_image_gradients_poisson_cosine_transform_2d(Pixel* gradient_x,
                                                                                 Pixel* gradient_y,
                                                                                 const uint width,
                                                                                 const uint height,
                                                                                 bool is_host_data)
{
    Pixel* divergence = CudaImageOperationsProcessor::divergence_2d(gradient_x, gradient_y,
                                                                 width, height, is_host_data);
    const uint depth = 1;
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


TGVDeshadeProcessor::MetricValuesHistory
TGVDeshadeProcessor::processTGV2L1DeshadeCuda_convergenceTest(
        ITKImage input_image,
        const Pixel lambda,
        const Pixel alpha0,
        const Pixel alpha1,
        const uint iteration_count,
        const uint check_iteration_interval,
        const ITKImage& mask,
        const bool set_negative_values_to_zero)
{
    Pixel* f = input_image.cloneToPixelArray();

    MetricValuesHistory metric_values_history;

  //  ITKImage r_before = ITKImage();
    ITKImage l_before = ITKImage();
    IterationCallback<Pixel> iteration_callback = [&l_before, &metric_values_history,
            &input_image, &mask, set_negative_values_to_zero]
            (uint iteration_index, uint iteration_count, Pixel* u_pixels, Pixel* v_x, Pixel* v_y, Pixel* v_z) {
        auto u = ITKImage(input_image.width, input_image.height, input_image.depth, u_pixels);
        auto l = ITKImage();
        auto r = deshade_poisson_cosine_transform(u, v_x, v_y, v_z,
                                                  input_image.width, input_image.height, input_image.depth,
                                                  mask, set_negative_values_to_zero, l, true);
        if(!l_before.isNull())
        {
            MetricValues metric_values;

            auto metric1 = normalized_cross_correlation(l, l_before);

            std::cout << "iteration " << iteration_index << ", metric1: " << metric1 << std::endl;

            metric_values.push_back(metric1);
            metric_values.push_back(time_tv(l, l_before));
     //       metric_values.push_back(normalized_cross_correlation(r, r_before));
    //        metric_values.push_back(time_tv(r, r_before));
            metric_values_history.push_back(metric_values);
        }
        l_before = l;
     //    r_before = r;

        return false;
    };

    Pixel* v_x, *v_y, *v_z;
    Pixel* u = tgv2_l1_deshade_launch<Pixel>(f,
                                             input_image.width, input_image.height, input_image.depth,
                                             lambda,
                                             iteration_count,
                                             check_iteration_interval,
                                             -1,
                                             iteration_callback,
                                             alpha0, alpha1,
                                             &v_x, &v_y, &v_z);
    delete[] f;
    delete[] u;
    delete[] v_x;
    delete[] v_y;
    if(input_image.depth > 1)
        delete[] v_z;

    return metric_values_history;
}

TGVDeshadeProcessor::MetricValues
TGVDeshadeProcessor::processTGV2L1DeshadeCuda_convergenceTestMetric(
        ITKImage input_image,
        const Pixel lambda,
        const Pixel alpha0,
        const Pixel alpha1,
        const uint iteration_count,
        const uint check_iteration_interval,
        const ITKImage& mask,
        const bool set_negative_values_to_zero)
{
    MetricValuesHistory metricValuesHistory = processTGV2L1DeshadeCuda_convergenceTest(
                input_image, lambda, alpha0, alpha1,
                iteration_count, check_iteration_interval,
                mask, set_negative_values_to_zero);

    MetricValues metricValuesTV;
    for(int m = 0; m < metricValuesHistory[0].size(); m++) // derivative/forward difference due to the iterations
    {
        MetricValue metricValueTV = 0;
        for(int iteration = 0; iteration < metricValuesHistory.size() - 2; iteration++)
        {
            MetricValues& metricValuesNextIteration = metricValuesHistory[iteration+1];
            MetricValues& metricValuesIteration = metricValuesHistory[iteration];

            MetricValue difference = metricValuesNextIteration[m] - metricValuesIteration[m];
            metricValueTV+= std::abs(difference);
        }
        metricValuesTV.push_back(metricValueTV);
    }
    return metricValuesTV;
}


void TGVDeshadeProcessor::processTGV2L1DeshadeCuda_convergenceTestToFile(
        ITKImage input_image,
        const Pixel lambda,
        const ITKImage& mask,
        const bool set_negative_values_to_zero,
        const Pixel alpha0,
        const Pixel alpha1,
        const uint check_iteration_count,
        std::string metric_file_name)
{
    MetricValues metricValues = processTGV2L1DeshadeCuda_convergenceTestMetric(
                input_image, lambda, alpha0, alpha1, check_iteration_count, 1, mask, set_negative_values_to_zero);

    std::ofstream stream;
    stream.open(metric_file_name);
    for(auto metric_value : metricValues)
    {
        stream << metric_value << std::endl;
    }
    stream.close();
}


void TGVDeshadeProcessor::processTGV2L1DeshadeCuda_convergenceOptimization(
        ITKImage input_image,
        const Pixel lambda,
        const ITKImage& mask,
        const bool set_negative_values_to_zero,
        ITKImage& denoised_image,
        ITKImage& shading_image,
        ITKImage& deshaded_image)
{
    const uint check_iteration_count = 10;

    // find decade of the step sizes
    const uint alpha_values_count = 11;

    MetricValue metricValueBefore = 0;

 //   const double alpha_values[alpha_values_count] = { 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1 };
    int converged_interval = 0;

    for(int i = 0; i < alpha_values_count; i++)
    {
        double alpha_value = std::pow(10, -i);

        std::cout << "                alpha_value: " << alpha_value << std::endl;

  //      auto alpha_value = alpha_values[i];
        MetricValues metricValues = processTGV2L1DeshadeCuda_convergenceTestMetric(
                    input_image, lambda, alpha_value, alpha_value,
                    check_iteration_count, 1, mask, set_negative_values_to_zero);
        MetricValue metricValue = metricValues[0];

        std::cout << "                metricValue: " << metricValue << std::endl;

        if(metricValue < metricValueBefore)
        {
            converged_interval = i - 1;
            std::cout << "                converged_interval: " << converged_interval << std::endl;
            break;
        }
        metricValueBefore = metricValue;
    }
    const double alpha1 = std::pow(10, -converged_interval);
    std::cout << "alpha1: " << alpha1 << std::endl;

    // find ratio of the step sizes
    const uint alpha_ratios_count = 9;
    const double alpha_ratios[alpha_ratios_count] = { 1, 1.5, 1.8, 2, 2.2, 2.3, 2.35, 2.4, 2.45 };
    // take maximum
    uint bestIndex = 0;
    MetricValue bestMetricValue = 0;
    for(int i = 0; i < alpha_ratios_count; i++)
    {
        auto alpha_ratio = alpha_ratios[i];
        auto alpha0 = alpha1 * alpha_ratio;
        MetricValues metricValues = processTGV2L1DeshadeCuda_convergenceTestMetric(
                    input_image, lambda, alpha0, alpha1,
                    check_iteration_count, 1, mask, set_negative_values_to_zero);

        MetricValue metricValue = metricValues[1];
        std::cout << "         alpha_ratios metrics: " << metricValue << std::endl;

        if(metricValue > bestMetricValue)
        {
            bestMetricValue = metricValue;
            bestIndex = i;
        }
    }

    const double alpha0 = alpha_ratios[bestIndex] * alpha1;
    std::cout << "alpha0: " << alpha0 << std::endl;
    std::cout << "alpha1: " << alpha1 << std::endl;

    const uint iteration_count = 500;
    processTGV2L1GPUCuda(
                input_image, lambda, alpha0, alpha1,
                iteration_count,
                -1,mask, set_negative_values_to_zero,
                denoised_image,
                shading_image,
                deshaded_image);
}

