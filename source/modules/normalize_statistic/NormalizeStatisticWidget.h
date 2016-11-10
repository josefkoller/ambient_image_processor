#ifndef NORMALIZESTATISTICWIDGET_H
#define NORMALIZESTATISTICWIDGET_H

#include "BaseModuleWidget.h"
#include "ImageViewWidget.h"

namespace Ui {
class NormalizeStatisticWidget;
}

class NormalizeStatisticWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    NormalizeStatisticWidget(QString title, QWidget *parent);
    ~NormalizeStatisticWidget();

protected:
    ITKImage processImage(ITKImage image);

private slots:
    void on_load_button_clicked();

    void on_performButton_clicked();

private:
    Ui::NormalizeStatisticWidget *ui;

    ImageViewWidget* reference_image_widget;
public:
    void registerModule(ImageWidget* image_widget);
};

#endif // NORMALIZESTATISTICWIDGET_H
