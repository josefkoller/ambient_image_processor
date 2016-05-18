#ifndef NONLOCALGRADIENTWIDGET_H
#define NONLOCALGRADIENTWIDGET_H

#include <QWidget>

#include <functional>

#include "NonLocalGradientProcessor.h"

#include "BaseModuleWidget.h"

namespace Ui {
class NonLocalGradientWidget;
}

class NonLocalGradientWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit NonLocalGradientWidget(QWidget *parent = 0);
    ~NonLocalGradientWidget();

private slots:
    void on_kernel_size_spinbox_valueChanged(int arg1);

    void on_sigma_spinbox_valueChanged(double arg1);

    void on_perform_button_clicked();

private:
    Ui::NonLocalGradientWidget *ui;
public:
    float getKernelSigma() const;
    uint getKernelSize() const;

protected:
    virtual ITKImage processImage(ITKImage image);
};

#endif // NONLOCALGRADIENTWIDGET_H
