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
