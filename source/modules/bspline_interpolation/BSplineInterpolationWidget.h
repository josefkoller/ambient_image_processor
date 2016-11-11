#ifndef BSPLINEINTERPOLATIONWIDGET_H
#define BSPLINEINTERPOLATIONWIDGET_H

#include <QWidget>
#include "BaseModuleWidget.h"
#include "ImageViewWidget.h"
#include "MaskWidget.h"

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

    MaskWidget::MaskFetcher mask_fetcher;

protected:
    virtual ITKImage processImage(ITKImage image);
private slots:
    void on_performButton_clicked();
};

#endif // BSPLINEINTERPOLATIONWIDGET_H
