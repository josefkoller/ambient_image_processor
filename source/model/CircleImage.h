#ifndef CIRCLEIMAGE_H
#define CIRCLEIMAGE_H

#include "Circle.h"
typedef unsigned int uint;
typedef float Pixel;

class CircleImage
{
public:
    Pixel* pixels;
    uint width;
    uint height;

    CircleImage(Circle circle, uint width, uint height);
};

#endif // CIRCLEIMAGE_H
