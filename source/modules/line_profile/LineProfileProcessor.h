#ifndef LINEPROFILEPROCESSOR_H
#define LINEPROFILEPROCESSOR_H

#include "ITKImage.h"

#include <vector>

#include "LineProfile.h"

class LineProfileProcessor
{
private:
    LineProfileProcessor();
public:
    static void intensity_profile(ITKImage image,
                                  LineProfile::Point point1,
                                  LineProfile::Point point2,
                                  std::vector<double>& intensities,
                                  std::vector<double>& distances);
};

#endif // LINEPROFILEPROCESSOR_H
