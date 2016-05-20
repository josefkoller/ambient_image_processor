#include "BaseModule.h"

BaseModule::BaseModule(QString title) : title(title)
{
}

QString BaseModule::getTitle() const
{
    return this->title;
}

void BaseModule::connectTo(BaseModule* other)
{
}

void BaseModule::registerModule(ImageWidget* image_widget)
{
    this->status_text_processor = [image_widget] (QString text) {
        emit image_widget->fireStatusTextChange(text);
    };
}

void BaseModule::setStatusText(QString text)
{
    if(this->status_text_processor != nullptr)
        this->status_text_processor(text);
}
