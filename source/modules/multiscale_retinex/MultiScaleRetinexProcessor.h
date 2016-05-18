#ifndef MULTISCALERETINEXPROCESSOR_H
#define MULTISCALERETINEXPROCESSOR_H

#include "ITKImage.h"
#include "MultiScaleRetinex.h"
#include <functional>

class MultiScaleRetinexProcessor
{
private:
    MultiScaleRetinexProcessor();
public:

    static ITKImage process(ITKImage image,
            std::vector<MultiScaleRetinex::Scale*> scales);
};

#endif // MULTISCALERETINEXPROCESSOR_H
