#ifndef BASEMODULE_H
#define BASEMODULE_H

#include <functional>
#include <QString>
#include "ImageWidget.h"

class BaseModule
{
public:
    BaseModule(QString title);

    QString getTitle() const;
    virtual void connectTo(BaseModule* other);

    virtual void registerModule(ImageWidget* image_widget);
private:
    QString title;

    std::function<void(QString)> status_text_processor;
protected:
    void setStatusText(QString text);
};

#endif // BASEMODULE_H
