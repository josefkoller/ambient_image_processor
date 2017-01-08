/*
    Ambient Image Processor - A tool to perform several imaging tasks
    
    Copyright (C) 2016 Josef Koller

    https://github.com/josefkoller/ambient_image_processor    
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#include <QApplication>
#include <QTextStream>
#include <QDir>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <algorithm>

#include "ITKImage.h"
#include "TGVKDeshadeDownsampledProcessor.h"
#include "TGVKDeshadeMaskedProcessor.h"
#include "TGVKDeshadeProcessor.h"
#include "TGVKDeshadeDownsampledNotMaskedProcessor.h"

#include "ImageViewWidget.h"
#include "LineProfileWidget.h"
#include "LineProfile.h"
#include "HistogramWidget.h"
#include "ImageInformationWidget.h"
#include "SliceControlWidget.h"

#include "cuda_host_helper.cuh"

typedef ITKImage::PixelType Pixel;

enum DeshadingMode {
    SubtractFromInput = 1,
    SubtractFromDenoised = 2,
    Color = 4,
    NotMasked = 8
};
inline DeshadingMode operator|(DeshadingMode a, DeshadingMode b) {
    return static_cast<DeshadingMode>(static_cast<int>(a) | static_cast<int>(b));
}

void writeColorImage(const ITKImage& image, std::string file_name, std::string template_file_name)
{
    QFile::copy(QString::fromStdString(template_file_name), QString::fromStdString(file_name));

    image.write_hsv(file_name);
}

void writeColorImageAndFigure(const ITKImage& image,
                std::string image_path,
                std::string template_file_name,
                std::string figure_with_overlays_path,
                ImageWidget& image_widget)
{
    image.write(image_path + ".mha");

    const auto color_file_path = image_path + ".png";
    writeColorImage(image, color_file_path, template_file_name);

    image_widget.setImage(image); // set grayscale information

    auto image_view = dynamic_cast<ImageViewWidget*>(image_widget.getModuleByName("Image View"));
    image_view->load_color_to_view_only(QString::fromStdString(color_file_path)); // set color info and add overlays

    image_view->save_file_with_overlays(QString::fromStdString(figure_with_overlays_path));
}

void writeImageAndFigure(const ITKImage& image,
                std::string image_path,
                std::string figure_with_overlays_path,
                ImageWidget& image_widget)
{
    image.write(image_path + ".mha");

    image_widget.setImage(image); // set grayscale information

    auto image_view = dynamic_cast<ImageViewWidget*>(image_widget.getModuleByName("Image View"));
    image_view->save_file_with_overlays(QString::fromStdString(figure_with_overlays_path));
}

void writeProfileLines(std::string file_path, ImageWidget& image_widget1, ImageWidget& image_widget2)
{
    auto line_profile2 = dynamic_cast<LineProfileWidget*>(image_widget2.getModuleByName("Line Profile"));
    line_profile2->save_to_file(QString::fromStdString(file_path));

    /*
    auto line_profile1 = dynamic_cast<LineProfileWidget*>(image_widget1.getModuleByName("Line Profile"));
    line_profile1->clearLineProfiles();
    */
}

void addProfileLines(ITKImage::Index point1, ITKImage::Index point2,
                     ImageWidget& image_widget)
{
    auto line_profile = dynamic_cast<LineProfileWidget*>(image_widget.getModuleByName("Line Profile"));
    LineProfile line;
    line.setPosition1(point1);
    line.setPosition2(point2);
    line_profile->clearLineProfiles();
    line_profile->addLineProfile(line);
}

void writeHistogram(const ImageWidget& image_widget, std::string path)
{
    auto histogram = dynamic_cast<HistogramWidget*>(image_widget.getModuleByName("Histogram"));

    histogram->calculateHistogramSync();
    histogram->write(QString::fromStdString(path));
}

void printMetrics(ImageWidget& image_widget1, ImageWidget& image_widget2, std::string text_file_path,
                  double computation_milliseconds, double memory_usage_bytes)
{
    auto histogram = dynamic_cast<HistogramWidget*>(image_widget1.getModuleByName("Histogram"));
    auto entropy1 = histogram->getEntropy();

    ITKImage::PixelType tv1, cv1;
    auto info = dynamic_cast<ImageInformationWidget*>(image_widget1.getModuleByName("Image Information"));
    info->getCVAndTV(cv1, tv1);

    QString log_text =  "\n\n";
    log_text += "entropy 1:\t" + QString::number(entropy1) + "\n";
    log_text += "cv 1:\t" + QString::number(cv1) + "\n";
    log_text += "tv 1:\t" + QString::number(tv1) + "\n";

    auto histogram2 = dynamic_cast<HistogramWidget*>(image_widget2.getModuleByName("Histogram"));
    auto entropy2 = histogram2->getEntropy();

    auto info2 = dynamic_cast<ImageInformationWidget*>(image_widget2.getModuleByName("Image Information"));
    ITKImage::PixelType tv2, cv2;
    info2->getCVAndTV(cv2, tv2);

    log_text +=  "\n";
    log_text += "entropy 2:\t" + QString::number(entropy2) + "\n";
    log_text += "cv 2:\t" + QString::number(cv2) + "\n";
    log_text += "tv 2:\t" + QString::number(tv2) + "\n";

    auto entropy_change = (entropy1 - entropy2) / entropy1 * 100;
    auto cv_change = (cv1 - cv2) / cv1 * 100;
    auto tv_change = (tv1 - tv2) / tv1 * 100;

    log_text +=  "\n";
    log_text += "entropy ratio:\t" + QString::number(entropy_change) + "% \n";
    log_text += "cv ratio:\t" + QString::number(cv_change) + "% \n";
    log_text += "tv ratio:\t" + QString::number(tv_change) + "% \n";

    log_text +=  "\n";
    log_text += "computation:\t" + QString::number(computation_milliseconds / 1000.0) + "s \n";
    log_text += "memory usage :\t" + QString::number(memory_usage_bytes / 1024.0 / 1024.0) + "MB \n";

    QFile text_file(QString::fromStdString(text_file_path));
    text_file.open(QIODevice::WriteOnly);
    QTextStream stream(&text_file);
    stream << log_text;

    std::cout << "saved " << text_file_path << std::endl;
}

void setMask(ImageWidget& image_widget, const ITKImage& mask)
{
    auto mask_widget = dynamic_cast<MaskWidget*>(image_widget.getModuleByName("Mask"));
    mask_widget->setMaskImage(mask);
}

void setSliceIndex(ImageWidget& image_widget, const uint slice_index)
{
    auto slice_control = dynamic_cast<SliceControlWidget*>(image_widget.getModuleByName("Slice Control"));
    slice_control->setSliceIndex(slice_index);
}


double gpuMemoryUsage()
{
    size_t free_byte;
    size_t total_byte;
    cudaCheckError( cudaMemGetInfo( &free_byte, &total_byte ) );

    return ((double)total_byte) - ((double)free_byte);
}

double runCUDAandReportMemoryUsage(std::function<void()> run_cuda,
                                   double& computation_milliseconds, double& memory_usage_bytes) {

    auto start = std::chrono::high_resolution_clock::now();

    memory_usage_bytes = gpuMemoryUsage();

    auto worker_thread = new std::thread([=]() {
        run_cuda();
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));

    memory_usage_bytes = gpuMemoryUsage() - memory_usage_bytes;
    worker_thread->join();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end-start;
    computation_milliseconds = elapsed.count();
}

void deshadeCuda(DeshadingMode mode, ITKImage input_image, const Pixel downsampling_factor, const Pixel lambda,
     const uint order, const std::vector<Pixel> alpha, const uint iteration_count, const ITKImage& mask,
     ITKImage& denoised_image, ITKImage& shading_image, ITKImage& deshaded_image)
{
    const bool set_negative_values_to_zero = true;
    const bool add_background_back = true;
    const uint paint_iteration_interval = 500;
    TGVKDeshadeDownsampledProcessor::IterationFinishedThreeImages iteration_finished_callback = []
            (uint iteration_index, uint iteration_count,
            ITKImage u, ITKImage l, ITKImage r) {
        std::cout << iteration_index << "/" << iteration_count << std::endl;
        return false; // do not stop
    };
    ITKImage div_v_image;
    const bool calculate_div_v = false;

    if((mode & SubtractFromInput) == SubtractFromInput)
    {
        if((mode & NotMasked) == NotMasked)
                TGVKDeshadeDownsampledNotMaskedProcessor::processTGVKL1Cuda(input_image,downsampling_factor,lambda, order, alpha.data(), iteration_count,
                                                                            mask, set_negative_values_to_zero, add_background_back,
                                                                            paint_iteration_interval, iteration_finished_callback,
                                                                            denoised_image, shading_image, deshaded_image, div_v_image,
                                                                            calculate_div_v);
        else
            TGVKDeshadeDownsampledProcessor::processTGVKL1Cuda(input_image,downsampling_factor,lambda, order, alpha.data(), iteration_count,
                                                               mask, set_negative_values_to_zero, add_background_back,
                                                               paint_iteration_interval, iteration_finished_callback,
                                                               denoised_image, shading_image, deshaded_image, div_v_image,
                                                               calculate_div_v);
    }
    else if((mode & SubtractFromDenoised) == SubtractFromDenoised)
    {
        const int cuda_block_dimension = -1; // take default
        if((mode & NotMasked) == NotMasked)
            TGVKDeshadeProcessor::processTGVKL1Cuda(input_image,lambda, order, alpha.data(), iteration_count,
                                                         mask, set_negative_values_to_zero, add_background_back,
                                                         paint_iteration_interval, iteration_finished_callback,
                                                         denoised_image, shading_image, deshaded_image, div_v_image,
                                                         calculate_div_v);
        else
            TGVKDeshadeMaskedProcessor::processTGVKL1Cuda(input_image,lambda, order, alpha.data(), iteration_count,
                                                          cuda_block_dimension,
                                                               mask, set_negative_values_to_zero, add_background_back,
                                                               paint_iteration_interval, iteration_finished_callback,
                                                               denoised_image, shading_image, deshaded_image, div_v_image,
                                                               calculate_div_v);
    }
}

void deshade(DeshadingMode mode, std::string data_root, std::string input_image_filename,
             ImageWidget& image_widget1, ImageWidget& image_widget2, const std::vector<Pixel> alpha,
             const Pixel downsampling_factor, const Pixel lambda, const uint iteration_count,
             ITKImage::Index line_profile_start, ITKImage::Index line_profile_end,
             std::string mask_filename = "")
{
    std::string input_file_path = data_root + input_image_filename;
    if(!QFile(QString::fromStdString(input_file_path)).exists())
        throw std::runtime_error("file does not exist: " + input_file_path);

    std::cout << "deshading: " << input_file_path << std::endl;

    const uint order = alpha.size();
    std::stringstream generated_directory_name_stream;
    generated_directory_name_stream << std::setprecision(1) << std::fixed;
    generated_directory_name_stream << "generated__lambda_" << lambda;

    if((mode & SubtractFromInput) == SubtractFromInput)
        generated_directory_name_stream << "__f_downsampling_" << downsampling_factor;

    generated_directory_name_stream << "__alpha";
    for(int i = 0; i < order; i++) {
        generated_directory_name_stream << "_" << alpha[i];
    }

    if((mode & NotMasked) == NotMasked)
        generated_directory_name_stream << "__not_masked";

    std::string generated_directory_name = generated_directory_name_stream.str();

    auto data_directory = QDir(QString::fromStdString(data_root));
    data_directory.mkpath(QString::fromStdString(generated_directory_name) + "/metaimages");
    data_directory.mkpath(QString::fromStdString(generated_directory_name) + "/figures");

    auto input_image = ITKImage::read(input_file_path);

    auto mask = mask_filename == "" ? ITKImage() : ITKImage::read(data_root + mask_filename);
    if(QFile(QString::fromStdString(data_root + mask_filename)).exists()) {
        setMask(image_widget1, mask);
        setMask(image_widget2, mask);
    } else {
        std::cout << "mask could not be loaded: " << data_root + mask_filename << std::endl;
    }

    setSliceIndex(image_widget1, line_profile_start[2]);
    setSliceIndex(image_widget2, line_profile_start[2]);

    ITKImage denoised_image, shading_image, deshaded_image;
    double computation_milliseconds, memory_usage_bytes;

    runCUDAandReportMemoryUsage([&]() {
        deshadeCuda(mode, input_image, downsampling_factor, lambda, order, alpha, iteration_count,
                    mask, denoised_image, shading_image, deshaded_image);
    }, computation_milliseconds, memory_usage_bytes);

    auto write = [&](const ITKImage& image, std::string image_path, std::string figure_with_overlays_path, ImageWidget& image_widget) {
        if((mode & Color) == Color)
            writeColorImageAndFigure(image, image_path, input_file_path, figure_with_overlays_path, image_widget);
        else
            writeImageAndFigure(image, image_path, figure_with_overlays_path, image_widget);
    };

    const std::string meta_images_root = data_root + generated_directory_name + "/metaimages/";
    const std::string figures_root = data_root + generated_directory_name + "/figures/";

    addProfileLines(line_profile_start, line_profile_end, image_widget1);
    image_widget1.setImage(input_image);
    image_widget2.setImage(deshaded_image);
    writeProfileLines(figures_root + "05_lines.png", image_widget1, image_widget2);

    writeHistogram(image_widget1, figures_root + "06_histogram_input.png");
    writeHistogram(image_widget2, figures_root + "06_histogram_deshaded.png");

    printMetrics(image_widget1, image_widget2, data_root + generated_directory_name + "/metrics.txt",
                 computation_milliseconds, memory_usage_bytes);

    write(input_image, meta_images_root + "01_input", figures_root + "01_input.png", image_widget1);
    write(deshaded_image, meta_images_root + "04_deshaded", figures_root + "04_deshaded.png", image_widget2);

    image_widget1.setImage(ITKImage());
    write(denoised_image, meta_images_root + "02_denoised", figures_root + "02_denoised.png", image_widget2);
    writeImageAndFigure(shading_image, meta_images_root + "03_shading", figures_root + "03_shading.png", image_widget2);

    image_widget2.setImage(ITKImage());
    std::cout << "finished deshading: " << input_file_path << std::endl;
}

std::vector<ITKImage::PixelType> getIncreasingAlpha(const uint order)
{
    std::vector<ITKImage::PixelType> alpha;
    for(int k = 0; k < order; k++)
        alpha.push_back(k + 1);
    return alpha;
}

void printHelp()
{
    std::cout << "Usage:" << std::endl;
    std::cout << "root_dir input mask mode line_start line_end downsampling_factor lambda iteration_count alpha" << std::endl;
    std::cout << std::endl;
    std::cout << "root_dir:\t" << "path to the data directory" << std::endl;
    std::cout << "input:\t" << "relative path from the root_dir to the input file" << std::endl;
    std::cout << "mask:\t" << "relative path to the mask file, if the file does not exists all voxels are inside of the mask" << std::endl;
    std::cout << "mode:\t" << "Combination of SubtractFromInput, SubtractFromDenoised, Color, NotMasked" << std::endl;
    std::cout << "\t" << "SubtractFromInput: the estimated bias is subtracted from the input image" << std::endl;
    std::cout << "\t" << "SubtractFromDenoised: the estimated bias is subtracted from the denoised image" << std::endl;
    std::cout << "\t" << "Color: the value channel (HSV) of the color input image is processed" << std::endl;
    std::cout << "\t" << "NotMasked: turn off mask performance improvement" << std::endl;
    std::cout << "line_start:\t" << "start index of the profile line included in the image plots" << std::endl;
    std::cout << "line_end:\t" << "end index of the profile line" << std::endl;
    std::cout << "downsampling_factor:\t" << "downsampling factor. 1, 0.5, 0.25..." << std::endl;
    std::cout << "lambda:\t" << "weight of the L1 data term. 1, 1.5, 2..." << std::endl;
    std::cout << "iteration_count:\t" << "1e3 - 1e5" << std::endl;
    std::cout << "alpha:\t" << "alpha parameter vector" << std::endl;
    std::cout << "\t" << "[1,2.1] for second order TGV" << std::endl;
    std::cout << "\t" << "[1,2,3] for third order TGV" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "\t data input.dcm mask.mha SubtractFromInput [0,0,0] [20,30,0] 1.0 1.0 1e4 [1,2]" << std::endl;
    std::cout << "\t data input.png non_existing_mask.mha SubtractFromInput+Color [0,0,0] [20,30,0] 1.0 1.0 1e4 [1,2,3]" << std::endl;
    std::cout << std::endl;
    std::cout << "Hints:" << std::endl;
    std::cout << "\t do not use NotMasked" << std::endl;
    std::cout << "\t do not include additional spaces" << std::endl;
    std::cout << "\t use a foreground mask" << std::endl;
    std::cout << "\t use increasing values for the alpha parameter" << std::endl;
}

ITKImage::Index parse3DIndex(const std::string& text)
{
    auto extractInteger = [=](uint from, uint to) {
        return std::stoi(text.substr(from, to - from + 1));
    };

    auto delimiter_position1 = text.find(',');
    auto delimiter_position2 = text.find(',', delimiter_position1 + 1);

    ITKImage::Index result;
    result[0] = extractInteger(1, delimiter_position1-1);
    result[1] = extractInteger(delimiter_position1+1, delimiter_position2-1);
    result[2] = extractInteger(delimiter_position2+1, text.length()-1);

    return result;
}

std::vector<Pixel> parseAlpha(const std::string& text)
{
    // std::cout << "parsing alpha: " << text << std::endl;

    std::vector<Pixel> alpha;

    const uint order = std::count(text.begin(), text.end(), ',') + 1;

    // std::cout << "order: " << order << std::endl;

    auto extractFloat = [&](uint from, uint to) {
        alpha.push_back(std::stof(text.substr(from, to - from + 1)));
    };
    uint start = 1;
    for(int i = 0; i < order; i++)
    {
        auto delimiter_position = text.find(',', start);

        if(i < order - 1)
            extractFloat(start, delimiter_position-1);
        else
            extractFloat(start, text.length()-2);
        start = delimiter_position + 1;
    }

    // std::cout << "parsed alpha element count: " << alpha.size() << std::endl;

    return alpha;
}

int main(int argc, char *argv[])
{
    std::cout << "started program: " << argv[0] << std::endl;

    if(argc > 1 && (argv[1] == std::string("--help")) || argc < 11) {
        printHelp();
        return 0;
    }

    const std::string samples_root_directory = argv[1];
    const std::string input_filename = argv[2];
    const std::string mask_filename = argv[3];

    const std::string mode_text = argv[4];

    std::cout << "mode: " << mode_text << std::endl;
    DeshadingMode mode = mode_text.find("SubtractFromInput") != std::string::npos ?
        SubtractFromInput : SubtractFromDenoised;
    if(mode_text.find("Color") != std::string::npos)  {
        mode = mode | Color;
    }
    if(mode_text.find("NotMasked") != std::string::npos)
        mode = mode | NotMasked;

    const auto line_profile_start = parse3DIndex(argv[5]);
    const auto line_profile_end = parse3DIndex(argv[6]);

    const Pixel downsampling_factor = std::stof(argv[7]);
    const Pixel lambda = std::stof(argv[8]);
    const uint iteration_count = std::stof(argv[9]);

    const std::vector<Pixel> alpha  = parseAlpha(argv[10]);

    QApplication application(argc, argv);

    ImageWidget image_widget1, image_widget2;
    image_widget1.setOutputWidget(&image_widget2);
    image_widget2.connectModule("Slice Control", &image_widget1);
    image_widget2.connectModule("Line Profile", &image_widget1);

    deshade(mode, samples_root_directory, input_filename, image_widget1, image_widget2,
            alpha, downsampling_factor, lambda, iteration_count,
            line_profile_start, line_profile_end, mask_filename);

    std::cout << "finished program: " << argv[0] << std::endl;
    return 0;
}
