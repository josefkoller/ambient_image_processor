#ifndef CONJUGATEGRADIENTWIDGET_H
#define CONJUGATEGRADIENTWIDGET_H

#include "BaseModuleWidget.h"

namespace Ui {
class ConjugateGradientWidget;
}

class ConjugateGradientWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    ConjugateGradientWidget(QString title, QWidget *parent);
    ~ConjugateGradientWidget();

private slots:
    void on_solve_poisson_button_clicked();

private:
    Ui::ConjugateGradientWidget *ui;

protected:
    ITKImage processImage(ITKImage image);
};

#endif // CONJUGATEGRADIENTWIDGET_H
