#ifndef TGV3WIDGET_H
#define TGV3WIDGET_H

#include "BaseModuleWidget.h"
#include "TGV3Processor.h"

namespace Ui {
class TGV3Widget;
}

class TGV3Widget : public BaseModuleWidget
{
    Q_OBJECT

public:
    TGV3Widget(QString title, QWidget *parent);
    ~TGV3Widget();

    void setIterationFinishedCallback(TGV3Processor::IterationFinished iteration_finished_callback);

    void registerModule(ImageWidget *image_widget);
protected:
    virtual ITKImage processImage(ITKImage image);
private slots:
    void on_perform_button_clicked();

    void on_stop_button_clicked();

private:
    Ui::TGV3Widget *ui;

    TGV3Processor::IterationFinished iteration_finished_callback;
    bool stop_after_next_iteration;
};

#endif // TGV3WIDGET_H
