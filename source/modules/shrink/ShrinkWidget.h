#ifndef SHRINKWIDGET_H
#define SHRINKWIDGET_H

#include <QWidget>

#include "BaseModuleWidget.h"

namespace Ui {
class ShrinkWidget;
}

class ShrinkWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit ShrinkWidget(QString title, QWidget *parent = 0);
    ~ShrinkWidget();
private slots:
    void on_shrink_button_clicked();

private:
    Ui::ShrinkWidget *ui;

protected:
    ITKImage processImage(ITKImage image);

public:
    void registerModule(ImageWidget* image_widget);
};

#endif // SHRINKWIDGET_H
