#ifndef OPENCVFILE_H
#define OPENCVFILE_H

#include "ITKImage.h"

#include <string>
using string = std::string;
#include <functional>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;

class OpenCVFile
{
private:
    OpenCVFile();

    typedef std::function<void(uint x, uint y, ITKImage::PixelType)> PixelWriter;
    typedef std::function<ITKImage::PixelType(uint x, uint y)> PixelReader;

    typedef std::function<void(std::vector<Mat>& channel)> ChannelsProcessor;
    static void read_hsv_and_process(string image_file_path, ChannelsProcessor processor);
public:
    static ITKImage read(string file_name);
    static void write_into_hsv_channel(const ITKImage& image, string file_name);
};

#endif // OPENCVFILE_H
