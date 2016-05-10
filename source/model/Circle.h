#ifndef CIRCLE_H
#define CIRCLE_H


class Circle
{
public:
    Circle(float inner_radius, float outer_radius, float factor) :
        inner_radius(inner_radius), outer_radius(outer_radius), factor(factor)
    {
    }

    float inner_radius;
    float outer_radius;
    float factor;
};

#endif // CIRCLE_H
