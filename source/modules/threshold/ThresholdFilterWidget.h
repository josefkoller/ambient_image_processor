#ifndef THRESHOLDFILTERWIDGET_H
#define THRESHOLDFILTERWIDGET_H

#include <QWidget>

#include "BaseModuleWidget.h"

namespace Ui {
class ThresholdFilterWidget;
}

class ThresholdFilterWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit ThresholdFilterWidget(QString title, QWidget *parent = 0);
    ~ThresholdFilterWidget();

private slots:
    void on_thresholdButton_clicked();

private:
    Ui::ThresholdFilterWidget *ui;

protected:
    ITKImage processImage(ITKImage image);
};

#endif // THRESHOLDFILTERWIDGET_H
