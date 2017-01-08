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
