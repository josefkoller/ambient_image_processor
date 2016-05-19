#ifndef UNSHARPMASKINGWIDGET_H
#define UNSHARPMASKINGWIDGET_H

#include <QWidget>

#include "BaseModuleWidget.h"

namespace Ui {
class UnsharpMaskingWidget;
}

class UnsharpMaskingWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit UnsharpMaskingWidget(QWidget *parent = 0);
    ~UnsharpMaskingWidget();

private slots:
    void on_perform_button_clicked();

    void on_factor_valueChanged(double arg1);

    void on_kernel_sigma_valueChanged(double arg1);

    void on_kernel_size_valueChanged(int arg1);

private:
    Ui::UnsharpMaskingWidget *ui;

protected:
    ITKImage processImage(ITKImage image);
};

#endif // UNSHARPMASKINGWIDGET_H
