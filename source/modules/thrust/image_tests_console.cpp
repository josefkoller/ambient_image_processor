

#include <iostream>
#include <string>

#include "ThrustImage.cuh"

typedef DeviceThrustImage ThrustThrustImage;

void print(ThrustThrustImage* ThrustImage, std::string title)
{
    std::cout << "ThrustImage: " << title << std::endl;
    for(int y = 0; y < ThrustImage->height; y++)
    {
        for(int x = 0; x < ThrustImage->width; x++)
        {
            std::cout << ThrustImage->getPixel(x,y) << "\t";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char *argv[])
{
    ThrustThrustImage* ThrustImage1 = new ThrustThrustImage(3,3);
    ThrustImage1->setPixel(0, 0, 1);
    ThrustImage1->setPixel(1, 0, 3);
    ThrustImage1->setPixel(2, 0, 7);

    ThrustImage1->setPixel(0, 1, 15);
    ThrustImage1->setPixel(1, 1, 25);
    ThrustImage1->setPixel(2, 1, 42);

    ThrustImage1->setPixel(0, 2, 85);
    ThrustImage1->setPixel(1, 2, 166);
    ThrustImage1->setPixel(2, 2, 512);

    print(ThrustImage1, "ThrustImage1");

    ThrustThrustImage* grad_x = ThrustImage1->clone_uninitialized();
    ThrustImage1->forward_difference_x(grad_x);
    print(grad_x, "grad_x");

    ThrustThrustImage* grad_y = ThrustImage1->clone_uninitialized();
    ThrustImage1->forward_difference_y(grad_y);
    print(grad_y, "grad_y");

    ThrustThrustImage* divergence = ThrustImage1->clone_uninitialized();
    ThrustThrustImage* grad_xx = ThrustImage1->clone_uninitialized();
    ThrustThrustImage* grad_yy = ThrustImage1->clone_uninitialized();
    ThrustThrustImage::divergence(grad_x, grad_y, grad_xx, grad_yy, divergence);


    print(grad_xx, "grad_xx");
    print(grad_yy, "grad_yy");
    print(divergence, "divergence");

    delete grad_yy;
    delete grad_xx;
    delete divergence;
    delete grad_x;
    delete grad_y;
    delete ThrustImage1;

    return 0;
}

