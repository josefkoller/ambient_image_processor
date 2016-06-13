#ifndef IMAGEVIEWCONTROLWIDGET_H
#define IMAGEVIEWCONTROLWIDGET_H

#include "BaseModuleWidget.h"

namespace Ui {
class ImageViewControlWidget;
}

class ImageViewControlWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    ImageViewControlWidget(QString title, QWidget *parent);
    ~ImageViewControlWidget();

private slots:
    void on_do_rescale_checkbox_toggled(bool checked);
    void on_do_multiply_checkbox_toggled(bool checked);

private:
    Ui::ImageViewControlWidget *ui;

signals:
    void doRescaleChanged(bool do_rescale);
    void doMultiplyChanged(bool do_multiply);
};

#endif // IMAGEVIEWCONTROLWIDGET_H