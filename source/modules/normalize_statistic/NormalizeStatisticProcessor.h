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

#ifndef NORMALIZESTATISTICPROCESSOR_H
#define NORMALIZESTATISTICPROCESSOR_H

#include "ITKImage.h"

class NormalizeStatisticProcessor
{
private:
    NormalizeStatisticProcessor();

public:
    static ITKImage equalizeMeanAddConstant(ITKImage image, ITKImage reference_image);
    static ITKImage equalizeMeanScale(ITKImage image, ITKImage reference_image);
    static ITKImage equalizeStandardDeviation(ITKImage image, ITKImage reference_image);
    static ITKImage equalizeMaxMin(ITKImage image, ITKImage reference_image);
};

#endif // NORMALIZESTATISTICPROCESSOR_H
