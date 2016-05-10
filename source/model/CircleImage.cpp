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

    const float max_distance = (circle.outer_radius - circle.inner_radius) * 0.5f;
    const float intensity_per_distance = (circle.factor - 1.0f) / max_distance;

    for(uint x = 0; x < width; x++)
        for(uint y = 0; y < height; y++)
        {
            float xf = ((float)x) / width * 2.0f - 1.0f + circle.x;
            float yf = ((float)y) / height * 2.0f - 1.0f + circle.y;
            float radius = sqrt(xf*xf + yf*yf);
            uint index = y*width + x;

            if(radius <= circle.inner_radius || radius >= circle.outer_radius)
            {
                this->pixels[index] = 1.0f;
                continue;
            }

            float inner_distance = std::abs(circle.inner_radius - radius);
            float outer_distance = std::abs(circle.outer_radius - radius);
            float distance = std::min(inner_distance, outer_distance);

            this->pixels[index] = distance * intensity_per_distance + 1.0f;
        }
}

