#ifndef TGVDESHADEINTEGRALMETRICPLOTWIDGET_H
#define TGVDESHADEINTEGRALMETRICPLOTWIDGET_H

#include "BaseModuleWidget.h"
#include "ImageViewWidget.h"

#include "TGVDeshadeProcessor.h"
#include "TGVDeshadeIntegralMetricPlotProcessor.h"

namespace Ui {
class TGVDeshadeIntegralMetricPlotWidget;
}

class TGVDeshadeIntegralMetricPlotWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    TGVDeshadeIntegralMetricPlotWidget(QString title, QWidget *parent);
    ~TGVDeshadeIntegralMetricPlotWidget();

    ImageViewWidget* shading_output_view;
    ImageViewWidget* denoised_output_view;
    ImageViewWidget* mask_view;

    TGVDeshadeIntegralMetricPlotProcessor::IterationFinished iteration_finished_callback;
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

    void handleMetricValuesChanged(std::vector<double> metricValues);

    void on_save_metric_plot_button_clicked();

private:
    Ui::TGVDeshadeIntegralMetricPlotWidget *ui;

    void plotMetricValues(TGVDeshadeIntegralMetricPlotProcessor::MetricValues metricValues);

signals:
    void fireMetricValuesChanged(std::vector<double> metricValues);
};

#endif // TGVDESHADEINTEGRALMETRICPLOTWIDGET_H
