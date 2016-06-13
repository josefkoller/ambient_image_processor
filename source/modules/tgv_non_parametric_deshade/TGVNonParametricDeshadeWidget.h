#ifndef TGVNONPARAMETRICDESHADEWIDGET_H
#define TGVNONPARAMETRICDESHADEWIDGET_H

#include "BaseModuleWidget.h"
#include "ImageViewWidget.h"

namespace Ui {
class TGVNonParametricDeshadeWidget;
}

class TGVNonParametricDeshadeWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    TGVNonParametricDeshadeWidget(QString title, QWidget *parent);
    ~TGVNonParametricDeshadeWidget();

    void registerModule(ImageWidget *image_widget);
private:
    Ui::TGVNonParametricDeshadeWidget *ui;

    ImageViewWidget* shading_output_view;
    ImageViewWidget* denoised_output_view;
    ImageViewWidget* mask_view;

protected:
    virtual ITKImage processImage(ITKImage image);
private slots:
    void on_load_mask_button_clicked();
    void on_save_denoised_button_clicked();
    void on_save_second_output_button_clicked();
    void on_perform_button_clicked();
};

#endif // TGVNONPARAMETRICDESHADEWIDGET_H
