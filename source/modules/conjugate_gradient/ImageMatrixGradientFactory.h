#ifndef IMAGEMATRIXGRADIENTFACTORY_H
#define IMAGEMATRIXGRADIENTFACTORY_H

#include "ImageMatrix.h"

class ImageMatrixGradientFactory
{
private:
    ImageMatrixGradientFactory();
public:
    typedef const unsigned int Dimension;

    template<typename Pixel>
    static void forwardDifferenceX(Dimension image_width, Dimension image_height, Dimension image_depth,
                                   ImageMatrix<Pixel>* matrix);
    template<typename Pixel>
    static void forwardDifferenceY(Dimension image_width, Dimension image_height, Dimension image_depth,
                                   ImageMatrix<Pixel>* matrix);
    template<typename Pixel>
    static void forwardDifferenceZ(Dimension image_width, Dimension image_height, Dimension image_depth,
                                   ImageMatrix<Pixel>* matrix);
    template<typename Pixel>
    static ImageMatrix<Pixel>* laplace(Dimension image_width,
                                                  Dimension image_height, Dimension image_depth);
};

#endif // IMAGEMATRIXGRADIENTFACTORY_H
