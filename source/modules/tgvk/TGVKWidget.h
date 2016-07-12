#ifndef TGVKWIDGET_H
#define TGVKWIDGET_H

#include "BaseModuleWidget.h"
#include "TGVKProcessor.h"

namespace Ui {
class TGVKWidget;
}

#include <QDoubleSpinBox>
#include <QVector>

namespace Ui {
class TGVKWidget;
}

class TGVKWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    TGVKWidget(QString title, QWidget *parent);
    ~TGVKWidget();

    void setIterationFinishedCallback(TGV3Processor::IterationFinished iteration_finished_callback);
protected:
    ITKImage processImage(ITKImage image);
    void registerModule(ImageWidget *image_widget);

    void addAlpha(uint index);
    void updateAlpha();
private slots:
    void on_perform_button_clicked();
    void on_stop_button_clicked();

    void on_order_spinbox_editingFinished();

protected:
    Ui::TGVKWidget *ui;

    QVector<QDoubleSpinBox*> alpha_spinboxes;
    TGVKProcessor::IterationFinished iteration_finished_callback;
    bool stop_after_next_iteration;
};

#endif // TGVKWIDGET_H
