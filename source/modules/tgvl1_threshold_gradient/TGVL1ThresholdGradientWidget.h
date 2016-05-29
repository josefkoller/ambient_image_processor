#ifndef TGVL1THRESHOLDGRADIENTWIDGET_H
#define TGVL1THRESHOLDGRADIENTWIDGET_H

#include <QWidget>

#include "BaseModuleWidget.h"

class ThresholdFilterWidget;
class NonLocalGradientWidget;
class ImageWidget;

namespace Ui {
class TGVL1ThresholdGradientWidget;
}

class TGVL1ThresholdGradientWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    TGVL1ThresholdGradientWidget(QString title, QWidget *parent);
    ~TGVL1ThresholdGradientWidget();

    virtual void registerModule(ImageWidget *image_widget);
private slots:
    void on_gradient_magnitude_button_clicked();

    void on_thresholded_gradient_magnitude_button_clicked();

private:
    Ui::TGVL1ThresholdGradientWidget *ui;

    ImageWidget *image_widget;

    QMap<QString, ITKImage> result_images;

    void setResult(QString submodule, ITKImage image);

protected:
    ITKImage processImage(ITKImage image);

};

#endif // TGVL1THRESHOLDGRADIENTWIDGET_H
