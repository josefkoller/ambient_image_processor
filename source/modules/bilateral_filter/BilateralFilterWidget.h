#ifndef BILATERALFILTERWIDGET_H
#define BILATERALFILTERWIDGET_H

#include <QWidget>

#include "BaseModuleWidget.h"

namespace Ui {
class BilateralFilterWidget;
}

class BilateralFilterWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit BilateralFilterWidget(QString title, QWidget *parent = 0);
    ~BilateralFilterWidget();

private slots:
    void on_perform_button_clicked();

private:
    Ui::BilateralFilterWidget *ui;

protected:
    ITKImage processImage(ITKImage image);
};

#endif // BILATERALFILTERWIDGET_H
