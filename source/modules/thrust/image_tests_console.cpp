

#include <iostream>
#include <string>

#include "Image.cuh"

typedef DeviceImage ThrustImage;

void print(ThrustImage* image, std::string title)
{
    std::cout << "image: " << title << std::endl;
    for(int y = 0; y < image->height; y++)
    {
        for(int x = 0; x < image->width; x++)
        {
            std::cout << image->getPixel(x,y) << "\t";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char *argv[])
{
    ThrustImage* image1 = new ThrustImage(3,3);
    image1->setPixel(0, 0, 1);
    image1->setPixel(1, 0, 3);
    image1->setPixel(2, 0, 7);

    image1->setPixel(0, 1, 15);
    image1->setPixel(1, 1, 25);
    image1->setPixel(2, 1, 42);

    image1->setPixel(0, 2, 85);
    image1->setPixel(1, 2, 166);
    image1->setPixel(2, 2, 512);

    print(image1, "image1");

    ThrustImage* grad_x = image1->clone_uninitialized();
    image1->forward_difference_x(grad_x);
    print(grad_x, "grad_x");

    ThrustImage* grad_y = image1->clone_uninitialized();
    image1->forward_difference_y(grad_y);
    print(grad_y, "grad_y");

    ThrustImage* divergence = image1->clone_uninitialized();
    ThrustImage* grad_xx = image1->clone_uninitialized();
    ThrustImage* grad_yy = image1->clone_uninitialized();
    ThrustImage::divergence(grad_x, grad_y, grad_xx, grad_yy, divergence);


    print(grad_xx, "grad_xx");
    print(grad_yy, "grad_yy");
    print(divergence, "divergence");

    delete grad_yy;
    delete grad_xx;
    delete divergence;
    delete grad_x;
    delete grad_y;
    delete image1;

    return 0;
}

