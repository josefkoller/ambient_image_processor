#ifndef IMAGEINFORMATIONWIDGET_H
#define IMAGEINFORMATIONWIDGET_H

#include <QWidget>

#include "BaseModuleWidget.h"

namespace Ui {
class ImageInformationWidget;
}

class ImageInformationWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit ImageInformationWidget(QString title, QWidget *parent = 0);
    ~ImageInformationWidget();

private:
    Ui::ImageInformationWidget *ui;

public:
    virtual void registerModule(ImageWidget* image_widget);

private slots:
    void collectInformation(ITKImage image);
    void on_copy_to_clipboard_button_clicked();
};

#endif // IMAGEINFORMATIONWIDGET_H
