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

#include "ShrinkFunctor.h"

ShrinkFunctor::ShrinkFunctor()
{

}


ShrinkFunctor::PixelType ShrinkFunctor::operator()(const PixelType& input)
{
    PixelType output;

    for(unsigned short d = 0; d < ITKImage::ImageDimension; d++)
    {
        const float input_value = input.GetElement(d);
        float output_value = 0;

        const float abs_input_value = std::abs(input_value);
        if(abs_input_value > 1e-6f && abs_input_value - lambda > 0)
        {
            output_value = input_value / abs_input_value * (abs_input_value - lambda);
        }

        output.SetElement(d, output_value);
    }
    return output;
}

void ShrinkFunctor::setLambda(float lambda)
{
    this->lambda = lambda;
}
