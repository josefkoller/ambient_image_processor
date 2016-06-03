#ifndef RESCALEINTENSITYWIDGET_H
#define RESCALEINTENSITYWIDGET_H

#include <QWidget>

#include "BaseModuleWidget.h"

namespace Ui {
class RescaleIntensityWidget;
}

class RescaleIntensityWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    RescaleIntensityWidget(QString title, QWidget *parent);
    ~RescaleIntensityWidget();

private slots:
    void on_perform_button_clicked();

private:
    Ui::RescaleIntensityWidget *ui;

protected:
    ITKImage processImage(ITKImage image);
};

#endif // RESCALEINTENSITYWIDGET_H
