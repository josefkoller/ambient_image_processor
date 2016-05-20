

#include <iostream>
#include <string>

#include "ThrustImage.cuh"

typedef DeviceThrustImage ThrustThrustImage;

void print(ThrustThrustImage* image, std::string title)
{
    std::cout << "ThrustImage: " << title << std::endl;
    for(int z = 0; z < image->depth; z++)
    {
        for(int y = 0; y < image->height; y++)
        {
            for(int x = 0; x < image->width; x++)
            {
                std::cout << image->getPixel(x,y,z) << "\t";
            }
            std::cout << std::endl;
        }
    }
}

int main(int argc, char *argv[])
{
    ThrustThrustImage* thrust_image1 = new ThrustThrustImage(3,3,1);
    thrust_image1->setPixel(0, 0, 1, 0);
    thrust_image1->setPixel(1, 0, 3, 0);
    thrust_image1->setPixel(2, 0, 7, 0);

    thrust_image1->setPixel(0, 1, 15, 0);
    thrust_image1->setPixel(1, 1, 25, 0);
    thrust_image1->setPixel(2, 1, 42, 0);

    thrust_image1->setPixel(0, 2, 85, 0);
    thrust_image1->setPixel(1, 2, 166, 0);
    thrust_image1->setPixel(2, 2, 512, 0);

    print(thrust_image1, "ThrustImage1");

    ThrustThrustImage* grad_x = thrust_image1->clone_uninitialized();
    thrust_image1->forward_difference_x(grad_x);
    print(grad_x, "grad_x");

    ThrustThrustImage* grad_y = thrust_image1->clone_uninitialized();
    thrust_image1->forward_difference_y(grad_y);
    print(grad_y, "grad_y");

    ThrustThrustImage* divergence = thrust_image1->clone_uninitialized();
    ThrustThrustImage* grad_xx = thrust_image1->clone_uninitialized();
    ThrustThrustImage* grad_yy = thrust_image1->clone_uninitialized();
    ThrustThrustImage::divergence(grad_x, grad_y, grad_xx, grad_yy, divergence);


    print(grad_xx, "grad_xx");
    print(grad_yy, "grad_yy");
    print(divergence, "divergence");

    delete grad_yy;
    delete grad_xx;
    delete divergence;
    delete grad_x;
    delete grad_y;
    delete thrust_image1;

    return 0;
}

