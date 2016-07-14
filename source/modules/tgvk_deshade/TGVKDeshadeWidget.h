#ifndef TGVKDESHADEWIDGET_H
#define TGVKDESHADEWIDGET_H

#include "BaseModuleWidget.h"
#include "TGVKDeshadeProcessor.h"
#include "ImageViewWidget.h"

#include <QVector>
#include <QDoubleSpinBox>

namespace Ui {
class TGVKDeshadeWidget;
}

class TGVKDeshadeWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    TGVKDeshadeWidget(QString title, QWidget *parent);
    ~TGVKDeshadeWidget();

private slots:
    void on_perform_button_clicked();
    void on_stop_button_clicked();
    void on_save_second_output_button_clicked();
    void on_load_mask_button_clicked();
    void on_save_denoised_button_clicked();

    void on_clear_mask_button_clicked();

    void on_order_spinbox_editingFinished();

    void on_save_div_v_button_clicked();

private:
    Ui::TGVKDeshadeWidget *ui;

protected:
    ImageViewWidget* shading_output_view;
    ImageViewWidget* denoised_output_view;
    ImageViewWidget* mask_view;
    ImageViewWidget* div_v_view;

    TGVKDeshadeProcessor::IterationFinishedThreeImages iteration_finished_callback;
    bool stop_after_next_iteration;
    QVector<QDoubleSpinBox*> alpha_spinboxes;

    virtual ITKImage processImage(ITKImage image);

    void addAlpha(uint index);
    void updateAlpha();
public:
    void setIterationFinishedCallback(TGVKDeshadeProcessor::IterationFinished iteration_finished_callback);

    void registerModule(ImageWidget *image_widget);
};

#endif // TGVKDESHADEWIDGET_H
