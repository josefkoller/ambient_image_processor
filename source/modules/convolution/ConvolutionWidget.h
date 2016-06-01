#ifndef CONVOLUTIONWIDGET_H
#define CONVOLUTIONWIDGET_H

#include <QWidget>

#include "BaseModuleWidget.h"

namespace Ui {
class ConvolutionWidget;
}

class ConvolutionWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    ConvolutionWidget(QString title, QWidget *parent);
    ~ConvolutionWidget();

private slots:
    void on_k12_valueChanged(double value);
    void on_k11_valueChanged(double value);
    void on_load_laplace_setting_button_clicked();

    void on_perform_button_clicked();

    void on_load_mean_setting_button_clicked();

private:
    Ui::ConvolutionWidget *ui;

protected:
    ITKImage processImage(ITKImage image);
};

#endif // CONVOLUTIONWIDGET_H
