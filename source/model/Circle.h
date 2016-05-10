#ifndef CIRCLE_H
#define CIRCLE_H


class Circle
{
public:
    Circle(float x, float y, float inner_radius, float outer_radius, float factor) :
        x(x), y(y), inner_radius(inner_radius), outer_radius(outer_radius), factor(factor)
    {
    }

    float inner_radius;
    float outer_radius;
    float factor;
    float x;
    float y;
};

#endif // CIRCLE_H
