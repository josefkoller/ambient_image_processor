#include "PixelIndex.h"

bool PixelIndex::isInside(PixelIndex size)
{
    return z >= 0 && z < size.z &&
           x >= 0 && x < size.x &&
           y >= 0 && y < size.y;
}

std::vector<PixelIndex> PixelIndex::collectNeighbours(PixelIndex size)
{
    std::vector<PixelIndex> indices;

    bool is_not_left = this->x > 0;
    bool is_not_top = this->y > 0;
    bool is_not_bottom = this->y < size.y - 1;
    bool is_not_right = this->x < size.x - 1;
    bool is_not_front = this->z > 0;
    bool is_not_back = this->z < size.z - 1;

    indices.push_back(*this);
    if(is_not_left)
    {
        indices.push_back(this->left());
        if(is_not_top)
            indices.push_back(this->leftTop());
        if(is_not_bottom)
            indices.push_back(this->leftBottom());
    }
    if(is_not_top)
        indices.push_back(this->top());
    if(is_not_bottom)
        indices.push_back(this->bottom());

    if(is_not_right)
    {
        indices.push_back(this->right());

        if(is_not_top)
            indices.push_back(this->rightTop());

        if(is_not_bottom)
            indices.push_back(this->rightBottom());
    }

    if(is_not_front)
    {
        indices.push_back(this->front());
        if(is_not_left)
        {
            indices.push_back(this->frontLeft());
            if(is_not_top)
                indices.push_back(this->frontLeftTop());
            if(is_not_bottom)
                indices.push_back(this->frontLeftBottom());
        }
        if(is_not_top)
            indices.push_back(this->frontTop());
        if(is_not_bottom)
            indices.push_back(this->frontBottom());

        if(is_not_right)
        {
            indices.push_back(this->frontRight());

            if(is_not_top)
                indices.push_back(this->frontRightTop());

            if(is_not_bottom)
                indices.push_back(this->frontRightBottom());
        }
    }

    if(is_not_back)
    {
        indices.push_back(this->back());
        if(is_not_left)
        {
            indices.push_back(this->backLeft());
            if(is_not_top)
                indices.push_back(this->backLeftTop());
            if(is_not_bottom)
                indices.push_back(this->backLeftBottom());
        }
        if(is_not_top)
            indices.push_back(this->backTop());
        if(is_not_bottom)
            indices.push_back(this->backBottom());

        if(is_not_right)
        {
            indices.push_back(this->backRight());

            if(is_not_top)
                indices.push_back(this->backRightTop());

            if(is_not_bottom)
                indices.push_back(this->backRightBottom());
        }
    }
    return indices;
}
