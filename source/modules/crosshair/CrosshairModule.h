#ifndef CROSSHAIRMODULE_H
#define CROSSHAIRMODULE_H

#include "BaseModule.h"

class ImageViewWidget;
class ImageWidget;

class CrosshairModule : public QObject, public BaseModule
{
    Q_OBJECT
private:
    ITKImage image;
public:
    CrosshairModule(QString title);

    void registerModule(ImageViewWidget* image_view_widget, ImageWidget* image_widget);

private slots:
    void mouseMoveOnImage(Qt::MouseButtons button, ITKImage::Index cursor_index);
};

#endif // CROSSHAIRMODULE_H
