/*
    Ambient Image Processor - A tool to perform several imaging tasks
    
    Copyright (C) 2016 Josef Koller

    https://github.com/josefkoller/ambient_image_processor    
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

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
    void setSliceIndex(uint slice_index);
private:
    Ui::SliceControlWidget *ui;

    ITKImage image;
    uint visible_slice_index;

    uint userSliceIndex() const;
    void setInputRanges();

private slots:
    void on_slice_slider_valueChanged(int slice_index);
    void on_slice_spinbox_valueChanged(int slice_index);

    void mouseWheelOnImage(int delta);
signals:
    void sliceIndexChanged(uint slice_index);
public slots:
    void connectedSliceControlChanged(uint slice_index);
};

#endif // SLICECONTROLWIDGET_H
