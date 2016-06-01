#ifndef TGVLAMBDASWIDGET_H
#define TGVLAMBDASWIDGET_H

#include <QWidget>

#include "BaseModuleWidget.h"
#include "ImageViewWidget.h"
#include "TGVLambdasProcessor.h"

namespace Ui {
class TGVLambdasWidget;
}

class TGVLambdasWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit TGVLambdasWidget(QString title, QWidget *parent = 0);
    ~TGVLambdasWidget();

    virtual void registerModule(ImageWidget* image_widget);

private slots:
    void on_load_button_clicked();

    void on_perform_button_clicked();

    void on_stop_button_clicked();

private:
    Ui::TGVLambdasWidget *ui;

    ITKImage lambdas_image;
    ImageViewWidget* lambdas_widget;

    bool stop_after_next_iteration;

    TGVLambdasProcessor::IterationFinished iteration_finished_callback;
public:
    void setIterationFinishedCallback(TGVLambdasProcessor::IterationFinished iteration_finished_callback);

protected:
    virtual ITKImage processImage(ITKImage image);
};

#endif // TGVLAMBDASWIDGET_H
