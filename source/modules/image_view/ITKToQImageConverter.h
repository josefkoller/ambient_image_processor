#ifndef ITKToQImageConverter_H
#define ITKToQImageConverter_H

#include <QImage>
#include "ITKImage.h"

class ITKToQImageConverter
{
public:
    static QImage* convert(ITKImage image,
                           ITKImage mask,
                           uint slice_index,
                           bool do_rescale,
                           bool do_multiply,
                           bool use_window);

    static void setWindowFrom(ITKImage::PixelType value);
    static void setWindowTo(ITKImage::PixelType value);

    static const QColor lower_window_color;
    static const QColor upper_window_color;
    static const QColor outside_mask_color;
private:
    static ITKImage::PixelType* window_from;
    static ITKImage::PixelType* window_to;
};

#endif // ITKToQImageConverter_H
