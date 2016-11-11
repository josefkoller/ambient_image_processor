#ifndef TGVDESHADEMASKEDWIDGET_H
#define TGVDESHADEMASKEDWIDGET_H

#include "BaseModuleWidget.h"
#include "ImageViewWidget.h"
#include "MaskWidget.h"

namespace Ui {
class TGVDeshadeMaskedWidget;
}

#include "TGVDeshadeMaskedProcessor.h"

class TGVDeshadeMaskedWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    TGVDeshadeMaskedWidget(QString title, QWidget *parent);
    ~TGVDeshadeMaskedWidget();

protected:
    Ui::TGVDeshadeMaskedWidget *ui;

    ImageViewWidget* shading_output_view;
    ImageViewWidget* denoised_output_view;
    MaskWidget::MaskFetcher mask_fetcher;

    TGVDeshadeMaskedProcessor::IterationFinishedThreeImages iteration_finished_callback;
    bool stop_after_next_iteration;
public:
    void setIterationFinishedCallback(TGVDeshadeMaskedProcessor::IterationFinished iteration_finished_callback);

    void registerModule(ImageWidget *image_widget);
protected:
    virtual ITKImage processImage(ITKImage image);
private slots:
    void on_perform_button_clicked();
    void on_stop_button_clicked();
    void on_save_second_output_button_clicked();
    void on_save_denoised_button_clicked();
};

#endif // TGVDESHADEMASKEDWIDGET_H
