#ifndef SLICECONTROLWIDGET_H
#define SLICECONTROLWIDGET_H

#include <QWidget>

#include "BaseModuleWidget.h"

namespace Ui {
class SliceControlWidget;
}

class SliceControlWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit SliceControlWidget(QString title, QWidget *parent = 0);
    ~SliceControlWidget();

    void registerModule(ImageWidget *image_widget);
    void connectTo(BaseModule *other);

    uint getVisibleSliceIndex() const { return this->visible_slice_index; };
private:
    Ui::SliceControlWidget *ui;

    ITKImage& image;
    uint visible_slice_index;

    void setSliceIndex(uint slice_index);
    uint userSliceIndex() const;
    void setInputRanges();

private slots:
    void on_slice_slider_valueChanged(int slice_index);
    void on_slice_spinbox_valueChanged(int slice_index);
    void on_slice_slider_sliderMoved(int position);

signals:
    void sliceIndexChanged(uint slice_index);
public slots:
    void connectedSliceControlChanged(uint slice_index);
};

#endif // SLICECONTROLWIDGET_H
