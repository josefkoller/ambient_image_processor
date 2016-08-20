#ifndef ORIGINSPACINGWIDGET_H
#define ORIGINSPACINGWIDGET_H

#include "BaseModuleWidget.h"

namespace Ui {
class OriginSpacingWidget;
}

class OriginSpacingWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    OriginSpacingWidget(QString title, QWidget *parent);
    ~OriginSpacingWidget();

    virtual void registerModule(ImageWidget* image_widget);

private slots:
    void on_performButton_clicked();

private:
    Ui::OriginSpacingWidget *ui;

protected:
    ITKImage processImage(ITKImage image);
};

#endif // ORIGINSPACINGWIDGET_H
