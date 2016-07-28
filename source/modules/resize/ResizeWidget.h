#ifndef RESIZEWIDGET_H
#define RESIZEWIDGET_H

#include "BaseModuleWidget.h"

namespace Ui {
class ResizeWidget;
}

class ResizeWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    ResizeWidget(QString title, QWidget *parent);
    ~ResizeWidget();

private slots:
    void on_perform_button_clicked();

private:
    Ui::ResizeWidget *ui;

protected:
    ITKImage processImage(ITKImage image);
};

#endif // RESIZEWIDGET_H
