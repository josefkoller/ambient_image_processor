
#include <string>
#include <iostream>

#include "ITKImage.h"
#include "TGVDeshadeProcessor.h"
#include "TGVDeshadeMaskedProcessor.h"

using string = std::string;


void printUsage()
{
    std::cout << "parameter: input_image_path lambda alpha0 alpha1 iteration_count cuda_block_dimension [mask] " <<
                 "output_denoised_path output_shading_path output_deshaded_path";
}

int process(
        ITKImage input_image,
        float lambda,
        float alpha0,
        float alpha1,
        int iteration_count,
        int cuda_block_dimension,
        ITKImage mask_image,
        string output_denoised_path,
        string output_shading_path,
        string output_deshaded_path)
{
    ITKImage denoised_image = ITKImage();
    ITKImage shading_image = ITKImage();
    ITKImage deshaded_image = ITKImage();

    if(mask_image.isNull())
    {
        TGVDeshadeProcessor::processTGV2L1GPUCuda(
                    input_image,
                    lambda,
                    alpha0,
                    alpha1,
                    iteration_count,
                    cuda_block_dimension,
                    mask_image,
                    true,
                    denoised_image,
                    shading_image,
                    deshaded_image);
    } else {
        deshaded_image = TGVDeshadeMaskedProcessor::processTGV2L1GPUCuda(
                    input_image,
                     lambda,
                     alpha0,
                     alpha1,
                     iteration_count,
                     cuda_block_dimension,
                     -1,
                     nullptr,
                     mask_image,
                     true,
                     true,
                     denoised_image,
                     shading_image);
    }

    denoised_image.write(output_denoised_path);
    shading_image.write(output_shading_path);
    deshaded_image.write(output_deshaded_path);

    return 0;
}


int process(
        string input_image_path,
        float lambda,
        float alpha0,
        float alpha1,
        int iteration_count,
        int cuda_block_dimension,
        string mask_path,
        string output_denoised_path,
        string output_shading_path,
        string output_deshaded_path)
{
    auto input_image = ITKImage::read(input_image_path);
    if(input_image.isNull())
    {
        std::cerr << "input image file does not exist: " << input_image_path << std::endl;
        return 2;
    }
    ITKImage mask_image = mask_path == "" ? ITKImage() : ITKImage::read(mask_path);
    if(mask_path != "" && mask_image.isNull())
    {
        std::cerr << "mask image file does not exist: " << mask_path << std::endl;
        return 3;
    }

    return process(input_image,
                   lambda,
                   alpha0,
                   alpha1,
                   iteration_count,
                   cuda_block_dimension,
                   mask_image,
                   output_denoised_path,
                   output_shading_path,
                   output_deshaded_path);
}


int main(int argc, char *argv[])
{
 //   std::cout << "started program: " << argv[0] << std::endl;
    if(argc < 9)
    {
        printUsage();
        return 1;
    }

    string input_image_path = argv[1];
    float lambda = std::stof(argv[2]);
    float alpha0 = std::stof(argv[3]);
    float alpha1 = std::stof(argv[4]);
    int iteration_count = std::stoi(argv[5]);
    int cuda_block_dimension = std::stoi(argv[6]);

    bool is_mask_given = argc == 11;
    string mask_path = "";
    string output_denoised_path = "";
    string output_shading_path = "";
    string output_deshaded_path = "";
    if(is_mask_given)
    {
        mask_path = argv[7];
        output_denoised_path = argv[8];
        output_shading_path = argv[9];
        output_deshaded_path = argv[10];
    }
    else
    {
        output_denoised_path = argv[7];
        output_shading_path = argv[8];
        output_deshaded_path = argv[9];
    }

    return process(input_image_path,
                   lambda,
                   alpha0,
                   alpha1,
                   iteration_count,
                   cuda_block_dimension,
                   mask_path,
                   output_denoised_path,
                   output_shading_path,
                   output_deshaded_path);
}

