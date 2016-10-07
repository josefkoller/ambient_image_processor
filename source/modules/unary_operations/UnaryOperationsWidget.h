#ifndef UNARYOPERATIONSWIDGET_H
#define UNARYOPERATIONSWIDGET_H

#include "BaseModuleWidget.h"

namespace Ui {
class UnaryOperationsWidget;
}

class UnaryOperationsWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    UnaryOperationsWidget(QString title, QWidget *parent);
    ~UnaryOperationsWidget();

private:
    Ui::UnaryOperationsWidget *ui;

    ITKImage rescale(ITKImage image);
protected:
    virtual ITKImage processImage(ITKImage image);
private slots:
    void on_perform_button_clicked();
};

#endif // UNARYOPERATIONSWIDGET_H
