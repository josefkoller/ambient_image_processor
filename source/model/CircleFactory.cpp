#include "CircleFactory.h"

#include <cstdlib>
#include <algorithm>

CircleFactory::CircleFactory()
{

}

float CircleFactory::random()
{
    return std::rand() / static_cast<float>(RAND_MAX);
}

Circle CircleFactory::createRandom()
{
     float r1 = 0; //random();
     float r2 = random();
     float inner_radius = std::min(r1,r2);
     float outer_radius = std::max(r1,r2);
     float factor = random()*0.9f + 0.1f;

     float x = 0.0f;
     float y = 0.0f;

     return Circle(x,y, inner_radius, outer_radius, factor);
}
