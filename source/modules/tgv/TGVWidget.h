#ifndef TGVWIDGET_H
#define TGVWIDGET_H

#include <QWidget>

namespace Ui {
class TGVWidget;
}

#include <thread>
#include "TGVProcessor.h"

#include "BaseModuleWidget.h"

class TGVWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit TGVWidget(QString title, QWidget *parent = 0);
    ~TGVWidget();

private slots:
    void on_perform_button_clicked();

    void on_stop_button_clicked();

private:
    Ui::TGVWidget *ui;
    TGVProcessor::IterationFinished iteration_finished_callback;

    bool stop_after_next_iteration;
public:
    void setIterationFinishedCallback(TGVProcessor::IterationFinished iteration_finished_callback);

    void registerModule(ImageWidget *image_widget);
protected:
    virtual ITKImage processImage(ITKImage image);
};

#endif // TGVWIDGET_H
