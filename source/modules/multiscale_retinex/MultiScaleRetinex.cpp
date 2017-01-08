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

#include "MultiScaleRetinex.h"
#include "MultiScaleRetinexScale.h"

#include <QLayout>

MultiScaleRetinex::MultiScaleRetinex()
{

}

void MultiScaleRetinex::addScaleTo(QWidget* frame)
{
    Scale* scale = new Scale();
    this->scales.push_back(scale);

    const unsigned int index = this->scales.size();
    auto removeCallback = [this, frame] (unsigned int scale_index) {
        if(this->scales.size() == 1)
            this->scales.clear();
        else
            this->scales.erase(this->scales.begin() + scale_index);

        // reindex all from scale_index .. end
        for(int i = scale_index; i < frame->layout()->count(); i++)
        {
            auto widget = dynamic_cast<MultiScaleRetinexScale*>(
                        frame->layout()->itemAt(i)->widget());
            if(widget != nullptr)
                widget->setIndex(i);
        }
    };

    new MultiScaleRetinexScale(frame, scale, index, removeCallback);
}
