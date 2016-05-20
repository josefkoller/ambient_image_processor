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
