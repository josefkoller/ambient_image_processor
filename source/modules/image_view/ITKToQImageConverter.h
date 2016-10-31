#ifndef ITKToQImageConverter_H
#define ITKToQImageConverter_H

#include <QImage>
#include "ITKImage.h"

class ITKToQImageConverter
{
public:
    static QImage* convert(ITKImage image, uint slice_index,
                           bool do_rescale,
                           bool do_multiply,
                           bool use_window);

    static void setWindowFrom(ITKImage::PixelType value);
    static void setWindowTo(ITKImage::PixelType value);
private:
    static ITKImage::PixelType* window_from;
    static ITKImage::PixelType* window_to;

    static const QColor lower_window_color;
    static const QColor upper_window_color;
};

#endif // ITKToQImageConverter_H
