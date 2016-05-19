#ifndef LINEPROFILEPROCESSOR_H
#define LINEPROFILEPROCESSOR_H

#include "ITKImage.h"

#include <vector>

class LineProfileProcessor
{
private:
    LineProfileProcessor();
public:
    static void intensity_profile(const ITKImage & image,
                                  int point1_x, int point1_y,
                                  int point2_x, int point2_y,
                                     std::vector<double>& intensities,
                                     std::vector<double>& distances);
};

#endif // LINEPROFILEPROCESSOR_H
