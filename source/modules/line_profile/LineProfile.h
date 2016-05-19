#ifndef LINEPROFILE_H
#define LINEPROFILE_H

#include <QPoint>
#include <QString>

struct LineProfile {
private:
    QPoint _position1;
    QPoint _position2;
    bool position1_is_set;
    bool position2_is_set;
public:
    LineProfile() : position1_is_set(false), position2_is_set(false)
    {}
    void setPosition1(QPoint position1) {
        this->_position1 = position1;
        this->position1_is_set = true;
    }
    void setPosition2(QPoint position2) {
        this->_position2 = position2;
        this->position2_is_set = true;
    }
    QPoint position1()
    {
        return this->_position1;
    }
    QPoint position2()
    {
        return this->_position2;
    }
    bool isSet()
    {
        return this->position1_is_set && this->position2_is_set;
    }

    QString text()
    {
        if(this->isSet())
        {
            return QString("%1 | %2  -  %3 | %4").arg(
                        QString::number(_position1.x()),
                        QString::number(_position1.y()),
                        QString::number(_position2.x()),
                        QString::number(_position2.y()) );
        }
        if(this->position1_is_set)
        {
            return "only position1 set";
        }
        if(this->position2_is_set)
        {
            return "only position2 set";
        }
        return "empty line";
    }
};

#endif // LINEPROFILE_H
