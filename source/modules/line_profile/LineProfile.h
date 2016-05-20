#ifndef LINEPROFILE_H
#define LINEPROFILE_H

#include <QString>

#include "ITKImage.h"


struct LineProfile {
public:
    typedef ITKImage::Index Point;

private:
    Point _position1;
    Point _position2;
    bool position1_is_set;
    bool position2_is_set;
public:

    LineProfile() : position1_is_set(false), position2_is_set(false)
    {}
    void setPosition1(Point position1) {
        this->_position1 = position1;
        this->position1_is_set = true;
    }
    void setPosition2(Point position2) {
        this->_position2 = position2;
        this->position2_is_set = true;
    }
    Point position1()
    {
        return this->_position1;
    }
    Point position2()
    {
        return this->_position2;
    }
    bool isSet()
    {
        return this->position1_is_set && this->position2_is_set;
    }

    QString text()
    {
        QString text;
        if(this->position1_is_set)
            text = ITKImage::indexToText(_position1);
        else
            text = "⦾";

        text += " ➞ ";

        if(this->position2_is_set)
            text += ITKImage::indexToText(_position2);
        else
            text += "⦿";

        return text;
    }
};

#endif // LINEPROFILE_H
