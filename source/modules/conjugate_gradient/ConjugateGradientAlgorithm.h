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

#ifndef CONJUGATEGRADIENTALGORITHM_H
#define CONJUGATEGRADIENTALGORITHM_H

template<typename Pixel>
class ImageMatrix;

class ConjugateGradientAlgorithm
{
private:
    typedef const unsigned int Dimension;
    ConjugateGradientAlgorithm();

public:
    template<typename Pixel>
    static void solveLinearEquationSystem(ImageMatrix<Pixel>* A,
                      Pixel* f, Pixel* x0,
                      const Pixel epsilon);

    template<typename Pixel>
    static void solvePoissonEquation(
            Pixel* f, Pixel* x0, Dimension image_width, Dimension image_height, Dimension image_depth,
            const Pixel epsilon);
};

#endif // CONJUGATEGRADIENTALGORITHM_H
