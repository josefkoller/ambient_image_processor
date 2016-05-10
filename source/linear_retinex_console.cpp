
#include <iostream>
#include <string>
typedef std::string string;

#include "ITKImageProcessor.h"
typedef ITKImageProcessor::ImageType Image;

int printHelp()
{
    std::cout << "Usage: ";
    std::cout << "[image file path] ";
    std::cout << "[alpha, float weighting gradient of illumination] ";
    std::cout << "[beta, float weighting gradient of reflectance] ";
    std::cout << "[pyramid levels, int] ";
    std::cout << "[iteration count factor, int] ";
    std::cout << "[output reflectance image path] ";
    std::cout << "[output illumination image path] " << std::endl;
    return 1;
}

void perform(std::string image_path, float alpha, float beta,
             int pyramid_levels,
             int iteration_count_factor,
             string output_reflectance_image_path,
             string output_illumination_image_path);

int main(int argc, char *argv[])
{
    if(argc != 8)
        return printHelp();

    string image_path = argv[1];
    float alpha = std::stof(argv[2]);
    float beta = std::stof(argv[3]);
    int pyramid_levels = std::stoi(argv[4]);
    int iteration_count_factor = std::stoi(argv[5]);

    string output_reflectance_image_path = argv[6];
    string output_illumination_image_path = argv[7];

    perform(image_path, alpha, beta, pyramid_levels,
            iteration_count_factor,
            output_reflectance_image_path, output_illumination_image_path);
    return 0;
}


void perform(std::string image_path, float alpha, float beta,
             int pyramid_levels,
             int iteration_count_factor,
             string output_reflectance_image_path,
             string output_illumination_image_path)
{
    Image::Pointer input_image = ITKImageProcessor::read(image_path);

    auto finished_callback = [=] (Image::Pointer R, Image::Pointer I)
    {
        ITKImageProcessor::write(R, output_reflectance_image_path);
        ITKImageProcessor::write(I, output_illumination_image_path);
    };

    ITKImageProcessor::removeSensorSensitivity( input_image,
                                                alpha, beta,
                                                pyramid_levels,
                                                iteration_count_factor,
                                                true,
                                                nullptr,
                                                finished_callback);
}
