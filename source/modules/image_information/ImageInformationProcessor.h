#ifndef IMAGEINFORMATIONPROCESSOR_H
#define IMAGEINFORMATIONPROCESSOR_H

#include "ITKImage.h"
#include <QMap>
#include <QString>

class ImageInformationProcessor
{
private:
    ImageInformationProcessor();
public:
    typedef QMap<QString, QString> InformationMap;
    static InformationMap collectInformation(ITKImage image);
    static double coefficient_of_variation(ITKImage itk_image);
};

#endif // IMAGEINFORMATIONPROCESSOR_H
