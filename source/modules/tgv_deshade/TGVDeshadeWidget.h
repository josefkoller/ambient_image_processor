#ifndef TGVDESHADEWIDGET_H
#define TGVDESHADEWIDGET_H

#include "BaseModuleWidget.h"

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

    TGVDeshadeProcessor::IterationFinished iteration_finished_callback;

    bool stop_after_next_iteration;
public:
    void setIterationFinishedCallback(TGVDeshadeProcessor::IterationFinished iteration_finished_callback);

    void registerModule(ImageWidget *image_widget);
protected:
    virtual ITKImage processImage(ITKImage image);
private slots:
    void on_perform_button_clicked();
    void on_stop_button_clicked();
};

#endif // TGVDESHADEWIDGET_H
