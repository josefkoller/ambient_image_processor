
#include <string>
#include <iostream>

#include "ITKImage.h"
#include "TGVDeshadeProcessor.h"

using string = std::string;


void printUsage()
{
    std::cout << "parameter: input_image_path lambda alpha0 alpha1 iteration_count [mask] " <<
                 "output_metric_file_name";
}

int process(
        ITKImage input_image,
        float lambda,
        float alpha0,
        float alpha1,
        int iteration_count,
        ITKImage mask_image,
        string output_metric_file_name)
{
    TGVDeshadeProcessor::processTGV2L1DeshadeCuda_convergenceTestToFile(
                input_image,
                lambda,
                mask_image,
                false,
                alpha0,
                alpha1,
                iteration_count,
                output_metric_file_name);

    return 0;
}


int process(
        string input_image_path,
        float lambda,
        float alpha0,
        float alpha1,
        int iteration_count,
        string mask_path,
        string output_metric_file_name)
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
                   mask_image,
                   output_metric_file_name);
}


int main(int argc, char *argv[])
{
    std::cout << "started program: " << argv[0] << std::endl;
    if(argc < 8)
    {
        printUsage();
        return 1;
    }

    string input_image_path = argv[1];
    float lambda = std::stof(argv[2]);
    float alpha0 = std::stof(argv[3]);
    float alpha1 = std::stof(argv[4]);
    int iteration_count = std::stoi(argv[5]);

    bool is_mask_given = argc == 8;
    string mask_path = "";
    string output_metric_file_name = "";
    if(is_mask_given)
    {
        mask_path = argv[6];
        output_metric_file_name = argv[7];
    }
    else
    {
        output_metric_file_name = argv[6];
    }

    std::cout << "parameter: " << std::endl <<
                 "input_image_path=" << input_image_path << std::endl <<
                 "lambda=" << lambda << std::endl <<
                 "alpha0=" << alpha0 << std::endl <<
                 "alpha1=" << alpha1 << std::endl <<
                 "iteration_count=" << iteration_count << std::endl <<
                 "mask=" << (is_mask_given ? mask_path : "none") << std::endl <<
                 "output_metric_file_name=" << output_metric_file_name;

    return process(input_image_path,
                   lambda,
                   alpha0,
                   alpha1,
                   iteration_count,
                   mask_path,
                   output_metric_file_name);
}

