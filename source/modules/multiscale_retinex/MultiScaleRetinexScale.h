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

#ifndef MULTISCALERETINEXSCALE_H
#define MULTISCALERETINEXSCALE_H

#include <functional>
#include <QWidget>
#include "MultiScaleRetinex.h"

namespace Ui {
class MultiScaleRetinexScale;
}

class MultiScaleRetinexScale : public QWidget
{
    Q_OBJECT

public:
    explicit MultiScaleRetinexScale(QWidget *frame, MultiScaleRetinex::Scale* scale,
                                     const unsigned int index,
                                    std::function<void(unsigned int scale_index)> removeCallback);
    ~MultiScaleRetinexScale();

    unsigned int getIndex() const;
    void setIndex(unsigned int index);
private:
    Ui::MultiScaleRetinexScale *ui;
    MultiScaleRetinex::Scale* scale;
    unsigned int index;
};

#endif // MULTISCALERETINEXSCALE_H
