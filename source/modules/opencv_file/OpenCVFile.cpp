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

         PixelReader pixel_reader = [&channel](uint x, uint y) { return channel.at<uchar>(y,x); };
         if(channel.depth() == CV_16U)
            pixel_reader = [&channel](uint x, uint y) { return channel.at<ushort>(y,x); };

         image.setEachPixel([pixel_reader](uint x, uint y, uint) {
             return pixel_reader(x,y);
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

         PixelWriter pixel_writer = [&channel](uint x, uint y, ITKImage::PixelType pixel) {
             channel.at<uchar>(y,x) = pixel; };
         if(channel.depth() == CV_16U)
            pixel_writer = [&channel](uint x, uint y, ITKImage::PixelType pixel) {
                channel.at<ushort>(y,x) = pixel; };

         image.foreachPixel([pixel_writer](uint x, uint y, uint, ITKImage::PixelType pixel) {
             pixel_writer(x, y, pixel);
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
