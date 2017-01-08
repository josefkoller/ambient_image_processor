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

#ifndef IMAGEMATRIXGRADIENTFACTORY_H
#define IMAGEMATRIXGRADIENTFACTORY_H

#include "ImageMatrix.h"

class ImageMatrixGradientFactory
{
private:
    ImageMatrixGradientFactory();
public:
    typedef const unsigned int Dimension;

    template<typename Pixel>
    static void forwardDifferenceX(Dimension image_width, Dimension image_height, Dimension image_depth,
                                   ImageMatrix<Pixel>* matrix);
    template<typename Pixel>
    static void forwardDifferenceY(Dimension image_width, Dimension image_height, Dimension image_depth,
                                   ImageMatrix<Pixel>* matrix);
    template<typename Pixel>
    static void forwardDifferenceZ(Dimension image_width, Dimension image_height, Dimension image_depth,
                                   ImageMatrix<Pixel>* matrix);
    template<typename Pixel>
    static ImageMatrix<Pixel>* laplace(Dimension image_width,
                                                  Dimension image_height, Dimension image_depth);
};

#endif // IMAGEMATRIXGRADIENTFACTORY_H
