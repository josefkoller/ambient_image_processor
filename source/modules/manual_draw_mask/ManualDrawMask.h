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

#ifndef MANUALDRAWMASK_H
#define MANUALDRAWMASK_H

#include "BaseModuleWidget.h"

namespace Ui {
class ManualDrawMask;
}

class ManualDrawMask : public BaseModuleWidget
{
    Q_OBJECT

public:
    ManualDrawMask(QString title, QWidget *parent = 0);
    ~ManualDrawMask();

protected:
    ITKImage processImage(ITKImage image);
    void registerModule(ImageWidget *image_widget);

private:
    Ui::ManualDrawMask *ui;

    bool is_drawing_mask;
    QVector<ITKImage::Index> polygon_points;
    const Qt::FillRule polygon_fill_rule;

    QPolygon createPolygon();
private slots:
    void mouseMoveOnImage(Qt::MouseButtons buttons, ITKImage::Index cursor_index);
    void mouseReleasedOnImage();
    void paintPolygon(QPixmap *pixmap);

    void on_startButton_clicked();

signals:
    void repaintImage();
};

#endif // MANUALDRAWMASK_H
