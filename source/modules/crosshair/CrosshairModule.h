#ifndef CROSSHAIRMODULE_H
#define CROSSHAIRMODULE_H

#include "BaseModule.h"
#include "ITKImage.h"

class CrosshairModule : public QObject, public BaseModule
{
    Q_OBJECT
private:
    ITKImage image;
public:
    CrosshairModule(QString title);

    virtual void registerModule(ImageWidget* image_widget);

private slots:
    void mouseMoveOnImage(Qt::MouseButtons button, QPoint position);
};

#endif // CROSSHAIRMODULE_H
