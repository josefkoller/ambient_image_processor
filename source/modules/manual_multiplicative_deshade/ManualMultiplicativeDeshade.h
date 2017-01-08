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

#ifndef MANUALMULTIPLICATIVEDESHADE_H
#define MANUALMULTIPLICATIVEDESHADE_H

#include <QWidget>

#include "BaseModuleWidget.h"

namespace Ui {
class ManualMultiplicativeDeshade;
}

class ManualMultiplicativeDeshade : public BaseModuleWidget
{
    Q_OBJECT

public:
    ManualMultiplicativeDeshade(QString title, QWidget *parent);
    ~ManualMultiplicativeDeshade();

    void registerModule(ImageWidget *image_widget);
private slots:
    void mousePressedOnImage(Qt::MouseButton button, ITKImage::Index position);

    void on_kernel_sigma_spinbox_editingFinished();
    void on_kernel_size_spinbox_editingFinished();
    void on_kernel_maximum_spinbox_editingFinished();

    void on_reset_shading_button_clicked();

private:
    Ui::ManualMultiplicativeDeshade *ui;

    ITKImage shading;
    ITKImage kernel;

    ITKImage::Index cursor_position;
    bool increase;

    void initShading();
    void generateKernel();

protected:
    ITKImage processImage(ITKImage image);
};

#endif // MANUALMULTIPLICATIVEDESHADE_H
