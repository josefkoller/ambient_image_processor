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
