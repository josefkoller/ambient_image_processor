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

#ifndef REGIONGROWINGSEGMENTATION_H
#define REGIONGROWINGSEGMENTATION_H

#include "ITKImage.h"

class RegionGrowingSegmentation
{
public:
    RegionGrowingSegmentation();

    typedef ITKImage::InnerITKImage::PixelType Pixel;
    typedef ITKImage::InnerITKImage::IndexType Position;
    typedef unsigned int uint;

    struct SeedPoint
    {
        Position position;
        Pixel tolerance;
        SeedPoint(Position position, Pixel tolerance) : position(position), tolerance(tolerance) {}
    };

    struct Segment
    {
        std::vector<SeedPoint> seed_points;
        std::string name;
        Segment(std::string name) : name(name) {}
    };
    typedef std::vector<Segment> Segments;
private:
    Segments segments;
    uint segment_counter_for_name;
public:
    Segment addSegment();
    void removeSegment(uint index);
    void setSegmentName(uint index, std::string name);
    void addSeedPoint(uint segment_index, SeedPoint position);
    void removeSeedPoint(uint segment_index, uint seed_point_index);
    void setSeedPointPosition(uint segment_index, uint seed_point_index, SeedPoint position);
    std::vector<SeedPoint> getSeedPointsOfSegment(uint segment_index) const;
    std::vector<std::vector<Position> > getSegments() const;
    Segments getSegmentObjects() const;
    void clear();
    Pixel getSeedPointTolerance(uint segment_index, uint seed_point_index) const;
    void setSeedPointTolerance(uint segment_index, uint point_index, Pixel tolerance);
};

#endif // REGIONGROWINGSEGMENTATION_H
