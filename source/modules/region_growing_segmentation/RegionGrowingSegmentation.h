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
