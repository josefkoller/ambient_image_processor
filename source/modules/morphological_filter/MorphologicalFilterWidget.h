#ifndef MORPHOLOGICALFILTERWIDGET_H
#define MORPHOLOGICALFILTERWIDGET_H

#include "BaseModuleWidget.h"

namespace Ui {
class MorphologicalFilterWidget;
}

class MorphologicalFilterWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    MorphologicalFilterWidget(QString title, QWidget *parent);
    ~MorphologicalFilterWidget();

private slots:
    void on_perform_button_clicked();

private:
    Ui::MorphologicalFilterWidget *ui;

protected:
    ITKImage processImage(ITKImage image);
};

#endif // MORPHOLOGICALFILTERWIDGET_H
