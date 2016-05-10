#ifndef CIRCLEFACTORY_H
#define CIRCLEFACTORY_H

#include "Circle.h"

class CircleFactory
{
private:
    CircleFactory();
    static float random();
public:
    static Circle createRandom();
};

#endif // CIRCLEFACTORY_H
