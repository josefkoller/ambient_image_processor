#ifndef MULTISCALERETINEX_H
#define MULTISCALERETINEX_H

#include <vector>
#include <QWidget>

class MultiScaleRetinex
{
public:
    struct Scale
    {
        float sigma;
        float weight;
    };

    std::vector<Scale*> scales;

    MultiScaleRetinex();
    void addScaleTo(QWidget* frame);
};

#endif // MULTISCALERETINEX_H
