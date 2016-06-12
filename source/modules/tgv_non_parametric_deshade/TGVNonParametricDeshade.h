#ifndef TGVNONPARAMETRICDESHADE_H
#define TGVNONPARAMETRICDESHADE_H

#include "TGVDeshadeWidget.h"

class TGVNonParametricDeshade : public TGVDeshadeWidget
{
    Q_OBJECT

public:
    TGVNonParametricDeshade(QString title, QWidget *parent);
    ~TGVNonParametricDeshade();

protected:
    ITKImage processImage(ITKImage image);
};

#endif // TGVNONPARAMETRICDESHADE_H
