#ifndef VECTOR3_H
#define VECTOR3_H

#include "ITKImage.h"

#include <math.h>

struct Vector3
{
    typedef ITKImage::PixelType ValueType;

    ValueType x;
    ValueType y;
    ValueType z;
    Vector3(ValueType x, ValueType y, ValueType z) : x(x), y(y), z(z) {}
    Vector3(ITKImage::IndexType index) : Vector3(index.x, index.y, index.z) {}
    Vector3(ITKImage::Index index) : Vector3(index[0], index[1], index[2]) {}

    Vector3& operator+=(ITKImage::Index index)
    {
        x += index[0];
        y += index[1];
        z += index[2];
        return *this;
    }
    Vector3 operator/(std::size_t size)
    {
        return Vector3(x / size,y / size,z / size);
    }
    Vector3 operator-(Vector3 vector)
    {
        return Vector3(x - vector.x, y - vector.y, z - vector.z);
    }
    ValueType length()
    {
        return std::sqrt(x*x + y*y + z*z);
    }
    Vector3& operator+=(Vector3 step)
    {
        x += step.x;
        y += step.y;
        z += step.z;
        return *this;
    }
    ITKImage::IndexType roundToIndex()
    {
        return {
            (uint)std::round(x),
            (uint)std::round(y),
            (uint)std::round(z)
        };
    }
    Vector3 operator*(ValueType factor)
    {
        return Vector3(x*factor, y*factor, z*factor);
    }
    Vector3 operator+(Vector3 step)
    {
        return Vector3(x+step.x, y+step.y, z+step.z);
    }
    Vector3 operator-()
    {
        return Vector3(-x, -y, -z);
    }

};

#endif // VECTOR3_H
