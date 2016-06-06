#ifndef BINARYOPERATIONSWIDGET_H
#define BINARYOPERATIONSWIDGET_H

#include <QWidget>

#include "BaseModuleWidget.h"
#include "ImageViewWidget.h"

namespace Ui {
class BinaryOperationsWidget;
}

class BinaryOperationsWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    BinaryOperationsWidget(QString title, QWidget *parent);
    ~BinaryOperationsWidget();

    void registerModule(ImageWidget *image_widget);
private slots:
    void on_load_button_clicked();

    void on_perform_button_clicked();

    void on_image2_offset_spinbox_valueChanged(double arg1);

    void on_image2_factor_spinbox_valueChanged(double arg1);

private:
    Ui::BinaryOperationsWidget *ui;
    ImageViewWidget* second_image_widget;

protected:
    virtual ITKImage processImage(ITKImage image);
};

#endif // BINARYOPERATIONSWIDGET_H
