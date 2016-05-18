#ifndef MULTISCALERETINEXWIDGET_H
#define MULTISCALERETINEXWIDGET_H

#include <QWidget>

#include "MultiScaleRetinex.h"

#include "BaseModuleWidget.h"

namespace Ui {
class MultiScaleRetinexWidget;
}

class MultiScaleRetinexWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit MultiScaleRetinexWidget(QWidget *parent = 0);
    ~MultiScaleRetinexWidget();

private slots:
    void on_addScaleButton_clicked();

    void on_calculate_button_clicked();

private:
    Ui::MultiScaleRetinexWidget *ui;

    MultiScaleRetinex multi_scale_retinex;

protected:
    virtual ITKImage processImage(ITKImage image);
};

#endif // MULTISCALERETINEXWIDGET_H
