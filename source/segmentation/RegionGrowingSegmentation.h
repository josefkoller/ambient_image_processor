#ifndef REGIONGROWINGSEGMENTATION_H
#define REGIONGROWINGSEGMENTATION_H

#include "../ITKImageProcessor.h"

class RegionGrowingSegmentation
{
public:
    RegionGrowingSegmentation();

    typedef ITKImageProcessor::ImageType::IndexType Position;
    typedef unsigned int uint;

    struct Segment
    {
        std::vector<Position> seed_points;
        std::string name;
        Segment(std::string name) : name(name) {}
    };
private:
    std::vector<Segment> segments;
    uint segment_counter_for_name;
public:
    Segment addSegment();
    void removeSegment(uint index);
    void setSegmentName(uint index, std::string name);
    void addSeedPoint(uint segment_index, Position position);
    void removeSeedPoint(uint segment_index, uint seed_point_index);
    void setSeedPointPosition(uint segment_index, uint seed_point_index, Position position);
    std::vector<Position> getSeedPointsOfSegment(uint segment_index) const;
    std::vector<std::vector<Position> > getSegments() const;
    std::vector<Segment> getSegmentObjects() const;
    void clear();
};

#endif // REGIONGROWINGSEGMENTATION_H
