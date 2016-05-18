#include "ShrinkFilter.h"

ShrinkFilter::ShrinkFilter()
{

}


void ShrinkFilter::setLambda(float lambda)
{
    this->GetFunctor().setLambda(lambda);
}
