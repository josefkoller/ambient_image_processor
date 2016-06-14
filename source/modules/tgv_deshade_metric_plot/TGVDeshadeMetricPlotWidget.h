#ifndef TGVDESHADEMETRICPLOTWIDGET_H
#define TGVDESHADEMETRICPLOTWIDGET_H

#include "BaseModuleWidget.h"
#include "ImageViewWidget.h"

#include "TGVDeshadeProcessor.h"
#include "TGVDeshadeMetricPlotProcessor.h"

namespace Ui {
class TGVDeshadeMetricPlotWidget;
}

class TGVDeshadeMetricPlotWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    TGVDeshadeMetricPlotWidget(QString title, QWidget *parent);
    ~TGVDeshadeMetricPlotWidget();

    ImageViewWidget* shading_output_view;
    ImageViewWidget* denoised_output_view;
    ImageViewWidget* mask_view;

    TGVDeshadeMetricPlotProcessor::IterationFinished iteration_finished_callback;
    bool stop_after_next_iteration;

public:
    void setIterationFinishedCallback(TGVDeshadeProcessor::IterationFinished iteration_finished_callback);

    void registerModule(ImageWidget *image_widget);
protected:
    virtual ITKImage processImage(ITKImage image);
private slots:
    void on_load_mask_button_clicked();

    void on_save_denoised_button_clicked();

    void on_save_second_output_button_clicked();

    void on_stop_button_clicked();

    void on_perform_button_clicked();

private:
    Ui::TGVDeshadeMetricPlotWidget *ui;

    void plotMetricValues(TGVDeshadeMetricPlotProcessor::MetricValues metricValues);
};

#endif // TGVDESHADEMETRICPLOTWIDGET_H
