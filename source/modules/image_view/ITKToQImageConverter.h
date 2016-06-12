#ifndef ITKToQImageConverter_H
#define ITKToQImageConverter_H

#include <QImage>
#include "ITKImage.h"

class ITKToQImageConverter
{
public:
    static QImage* convert(ITKImage image, uint slice_index,
                           bool do_rescale,
                           bool do_multiply);

    static void setWindowFrom(ITKImage::PixelType value);
    static void setWindowTo(ITKImage::PixelType value);
private:
    static ITKImage::PixelType* window_from;
    static ITKImage::PixelType* window_to;
};

#endif // ITKToQImageConverter_H
