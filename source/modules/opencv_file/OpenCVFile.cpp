#include "OpenCVFile.h"

#include <iostream>

OpenCVFile::OpenCVFile()
{
}

void OpenCVFile::read_hsv_and_process(string image_file_path,
                                  ChannelsProcessor processor)
{
    Mat image = imread(image_file_path, CV_LOAD_IMAGE_UNCHANGED);

    if(! image.data )
    {
        std::cout <<  "Could not open or find the image: " << image_file_path << std::endl ;
        return;
    }
    std::vector<Mat> channels;
    split(image, channels);

    Mat& data = image;
    if(channels.size() > 1) // is color... take the HSV value channel only
    {
        Mat image_hsv;
        cvtColor(image, image_hsv, COLOR_BGR2HSV);
        split(image_hsv, channels);
        data = channels[2];
    }
    processor(channels);
}

ITKImage OpenCVFile::read(string image_file_path)
{
    auto image = ITKImage();
    read_hsv_and_process(image_file_path, [&image] (std::vector<Mat>& channels) {
         const Mat& channel = channels.size() > 1 ? channels[2] : channels[0];
         image = ITKImage(channel.cols, channel.rows, 1);
         image.setEachPixel([&channel](uint x, uint y, uint) {
             return channel.at<uchar>(y,x);
         });
    });

    return image;
}

void OpenCVFile::write_into_hsv_channel(const ITKImage& image, string file_name)
{
    read_hsv_and_process(file_name, [&image, file_name] (std::vector<Mat>& channels) {
         Mat& channel = channels.size() > 1 ? channels[2] : channels[0];
         if(image.width != channel.cols ||
                 image.height != channel.rows)
         {
             std::cerr << "image dimension mismatch" << std::endl;
             return;
         }

         image.foreachPixel([&channel](uint x, uint y, uint, ITKImage::PixelType pixel) {
             channel.at<uchar>(y,x) = pixel;
         });

         Mat& image_to_save = channel;
         Mat merged_image;
         Mat rgb_image;
         if(channels.size() > 1)
         {
             merge(channels, merged_image);
             cvtColor(merged_image, rgb_image, COLOR_HSV2BGR);
             image_to_save = rgb_image;
         }

         imwrite(file_name, image_to_save);
    });
}
