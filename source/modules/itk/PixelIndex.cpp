/*
    Ambient Image Processor - A tool to perform several imaging tasks
    
    Copyright (C) 2016 Josef Koller

    https://github.com/josefkoller/ambient_image_processor    
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "PixelIndex.h"

PixelIndex::PixelIndex(uint linear_index, PixelIndex size)
{
    z = linear_index / (size.x*size.y);
    int index_rest = linear_index - z * (size.x*size.y);
    y = index_rest / size.x;
    index_rest = index_rest - y * size.x;
    x = index_rest;
}

bool PixelIndex::isInside(PixelIndex size) const
{
    return z >= 0 && z < size.z &&
           x >= 0 && x < size.x &&
           y >= 0 && y < size.y;
}

PixelIndex::uint PixelIndex::linearIndex(PixelIndex size) const
{
    return z * size.x*size.y + x + y * size.x;
}

std::vector<PixelIndex> PixelIndex::collectNeighboursInSlice(PixelIndex size) const
{
    std::vector<PixelIndex> indices;

    bool is_not_left = this->x > 0;
    bool is_not_top = this->y > 0;
    bool is_not_bottom = this->y < size.y - 1;
    bool is_not_right = this->x < size.x - 1;

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
    return indices;
}

std::vector<PixelIndex> PixelIndex::collectNeighbours(PixelIndex size) const
{
    bool is_not_left = this->x > 0;
    bool is_not_top = this->y > 0;
    bool is_not_bottom = this->y < size.y - 1;
    bool is_not_right = this->x < size.x - 1;

    bool is_not_front = this->z > 0;
    bool is_not_back = this->z < size.z - 1;

    auto indices = this->collectNeighboursInSlice(size);
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
