#ifndef TGVDESHADEWIDGET_H
#define TGVDESHADEWIDGET_H

#include "BaseModuleWidget.h"
#include "ImageViewWidget.h"

namespace Ui {
class TGVDeshadeWidget;
}

#include "TGVDeshadeProcessor.h"

class TGVDeshadeWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    TGVDeshadeWidget(QString title, QWidget *parent);
    ~TGVDeshadeWidget();

private:
    Ui::TGVDeshadeWidget *ui;

    ImageViewWidget* shading_output_view;
    ImageViewWidget* denoised_output_view;
    ImageViewWidget* mask_view;

    TGVDeshadeProcessor::IterationFinishedThreeImages iteration_finished_callback;
    bool stop_after_next_iteration;
public:
    void setIterationFinishedCallback(TGVDeshadeProcessor::IterationFinished iteration_finished_callback);

    void registerModule(ImageWidget *image_widget);
protected:
    virtual ITKImage processImage(ITKImage image);
private slots:
    void on_perform_button_clicked();
    void on_stop_button_clicked();
    void on_save_second_output_button_clicked();
    void on_load_mask_button_clicked();
    void on_save_denoised_button_clicked();
};

#endif // TGVDESHADEWIDGET_H
