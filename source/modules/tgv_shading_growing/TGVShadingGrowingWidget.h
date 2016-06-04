#ifndef TGVSHADINGGROWINGWIDGET_H
#define TGVSHADINGGROWINGWIDGET_H

#include <QWidget>

#include "BaseModuleWidget.h"
#include "TGVShadingGrowingProcessor.h"

namespace Ui {
class TGVShadingGrowingWidget;
}

class TGVShadingGrowingWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    TGVShadingGrowingWidget(QString title, QWidget *parent);
    ~TGVShadingGrowingWidget();

    void registerModule(ImageWidget *image_widget);

    typedef TGVShadingGrowingProcessor::IterationFinished IterationFinished;
    void setIterationFinishedCallback(IterationFinished iteration_finished_callback);
private slots:
    void on_perform_button_clicked();

    void on_stop_button_clicked();

private:
    Ui::TGVShadingGrowingWidget *ui;

    IterationFinished iteration_finished_callback;
    bool stop_after_next_iteration;
protected:
    ITKImage processImage(ITKImage image);
};

#endif // TGVSHADINGGROWINGWIDGET_H
