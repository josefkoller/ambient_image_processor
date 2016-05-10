#include "CircleImage.h"

#include <cmath>

#include <iostream>

CircleImage::CircleImage(Circle circle, uint width, uint height) :
    width(width), height(height)
{
    uint pixel_count = width * height;
    this->pixels = new Pixel[pixel_count];

    /*
    std::cout << "factor: " << circle.factor << std::endl;
    std::cout << "inner: " << circle.inner_radius << std::endl;
    std::cout << "outer: " << circle.outer_radius << std::endl;
    */

    for(uint x = 0; x < width; x++)
        for(uint y = 0; y < height; y++)
        {
            float xf = ((float)x) / width * 2.0f - 1.0f;
            float yf = ((float)y) / height * 2.0f - 1.0f;
            float radius = sqrt(xf*xf + yf*yf);
            uint index = y*width + x;

            this->pixels[index] = (radius > circle.inner_radius &&
               radius < circle.outer_radius) ? circle.factor : 1.0f;
        }
}

