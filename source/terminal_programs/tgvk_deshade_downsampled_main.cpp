
#include <string>
#include <iostream>
#include <chrono>
#include <ctime>

#include "ITKImage.h"
#include "TGVKDeshadeDownsampledProcessor.h"

using string = std::string;
typedef unsigned int uint;


void printUsage()
{
    std::cout << "parameter: input_image_path f_downsampling order lambda iteration_count [mask] " <<
                 "output_image_prefix";
}
int process(
        ITKImage input_image,
        float f_downsampling,
        uint order,
        float lambda,
        int iteration_count,
        ITKImage mask_image,
        string output_image_prefix)
{
    ITKImage denoised_image = ITKImage();
    ITKImage shading_image = ITKImage();
    ITKImage deshaded_image = ITKImage();
    ITKImage div_v_image = ITKImage();

    ITKImage::PixelType* alpha = new ITKImage::PixelType[order];
    for(int i = 0; i < order; i++)
      alpha[i] = i + 1;

    uint paint_iteration_interval = iteration_count + 1;


    typedef std::chrono::time_point<std::chrono::system_clock> Timestamp;
    Timestamp start = std::chrono::system_clock::now();

    std::cout << "mask image is null: " << mask_image.isNull() << std::endl;

    TGVKDeshadeDownsampledProcessor::processTGVKL1Cuda(
                input_image,
                f_downsampling,
                lambda,
                order,
                alpha,
                iteration_count,
                mask_image,
                true,
                true,
                paint_iteration_interval,
                nullptr,
                denoised_image,
                shading_image,
                deshaded_image,
                div_v_image, false);

    Timestamp end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    std::string output_shading_path = output_image_prefix + "_shading.mha";
    std::string output_deshaded_path = output_image_prefix + "_deshaded.mha";

    shading_image.write(output_shading_path);
    deshaded_image.write(output_deshaded_path);

    return 0;
}

int process(
        string input_image_path,
        float f_downsampling,
        uint order,
        float lambda,
        int iteration_count,
        string mask_path,
        string output_image_prefix)
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
                   f_downsampling,
                   order,
                   lambda,
                   iteration_count,
                   mask_image,
                   output_image_prefix);
}


int main(int argc, char *argv[])
{
    std::cout << "started program: " << argv[0] << std::endl;
    if(argc < 7)
    {
        printUsage();
        return 1;
    }

    string input_image_path = argv[1];
    float f_downsampling = std::stof(argv[2]);
    uint order = std::stoi(argv[3]);
    float lambda = std::stof(argv[4]);
    int iteration_count = std::stoi(argv[5]);

    bool is_mask_given = argc == 8;
    string mask_path = "";
    string output_image_prefix = "";
    if(is_mask_given)
    {
        mask_path = argv[6];
        output_image_prefix = argv[7];
    }
    else
    {
        output_image_prefix = argv[6];
    }

    std::cout << "input_image: " << input_image_path << std::endl;
    std::cout << "f_downsampling: " << f_downsampling << std::endl;
    std::cout << "order: " << order << std::endl;
    std::cout << "lambda: " << lambda << std::endl;
    std::cout << "iteration_count: " << iteration_count << std::endl;
    std::cout << "mask_path: " << mask_path << std::endl;
    std::cout << "output_image_prefix: " << output_image_prefix << std::endl;

    return process(input_image_path,
                   f_downsampling,
                   order,
                   lambda,
                   iteration_count,
                   mask_path,
                   output_image_prefix);
}

