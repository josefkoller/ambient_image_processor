#ifndef NORMALIZESTATISTICPROCESSOR_H
#define NORMALIZESTATISTICPROCESSOR_H

#include "ITKImage.h"

class NormalizeStatisticProcessor
{
private:
    NormalizeStatisticProcessor();

public:
    static ITKImage equalizeMean(ITKImage image, ITKImage reference_image);
    static ITKImage equalizeStandardDeviation(ITKImage image, ITKImage reference_image);
    static ITKImage equalizeMaxMin(ITKImage image, ITKImage reference_image);
};

#endif // NORMALIZESTATISTICPROCESSOR_H
