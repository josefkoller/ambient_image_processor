#ifndef BSPLINEINTERPOLATIONWIDGET_H
#define BSPLINEINTERPOLATIONWIDGET_H

#include <QWidget>
#include "BaseModuleWidget.h"
#include "ImageViewWidget.h"

namespace Ui {
class BSplineInterpolationWidget;
}

class BSplineInterpolationWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    BSplineInterpolationWidget(QString title, QWidget *parent);
    ~BSplineInterpolationWidget();

    void registerModule(ImageWidget *image_widget);
private:
    Ui::BSplineInterpolationWidget *ui;

    ImageViewWidget* mask_view;

protected:
    virtual ITKImage processImage(ITKImage image);
private slots:
    void on_performButton_clicked();
    void on_load_mask_button_clicked();
    void on_clear_mask_button_clicked();
};

#endif // BSPLINEINTERPOLATIONWIDGET_H
