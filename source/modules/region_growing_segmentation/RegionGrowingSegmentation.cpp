#include "RegionGrowingSegmentation.h"

RegionGrowingSegmentation::RegionGrowingSegmentation()
    : segment_counter_for_name(0)
{

}

RegionGrowingSegmentation::Segment RegionGrowingSegmentation::addSegment()
{
    std::string name = "Segment" + std::to_string(segment_counter_for_name++);
    Segment segment(name);
    this->segments.push_back(segment);
    return segment;
}

void RegionGrowingSegmentation::removeSegment(uint index)
{
    this->segments.erase(this->segments.begin() + index);
}

void RegionGrowingSegmentation::removeSeedPoint(uint segment_index, uint seed_point_index)
{
    auto& seed_points = this->segments[segment_index].seed_points;
    seed_points.erase(seed_points.begin() + seed_point_index);
}

void RegionGrowingSegmentation::setSegmentName(uint index, std::string name)
{
    this->segments[index].name = name;
}

void RegionGrowingSegmentation::addSeedPoint(uint segment_index, SeedPoint position)
{
    this->segments[segment_index].seed_points.push_back(position);
}

std::vector<RegionGrowingSegmentation::SeedPoint> RegionGrowingSegmentation::getSeedPointsOfSegment(uint segment_index) const
{
    return this->segments[segment_index].seed_points;
}

std::vector<std::vector<RegionGrowingSegmentation::Position> > RegionGrowingSegmentation::getSegments() const
{
    std::vector<std::vector<RegionGrowingSegmentation::Position> > segments;
    for(Segment segment : this->segments)
    {
        std::vector<RegionGrowingSegmentation::Position> segment_points;
        for(auto point : segment.seed_points)
            segment_points.push_back(point.position);
        segments.push_back(segment_points);
    }
    return segments;
}

std::vector<RegionGrowingSegmentation::Segment> RegionGrowingSegmentation::getSegmentObjects() const
{
    return this->segments;
}

void RegionGrowingSegmentation::clear()
{
    this->segments.clear();
}

RegionGrowingSegmentation::Pixel RegionGrowingSegmentation::getSeedPointTolerance(uint segment_index,
                                                                                  uint seed_point_index) const
{
    return this->segments[segment_index].seed_points[seed_point_index].tolerance;
}

void RegionGrowingSegmentation::setSeedPointTolerance(uint segment_index, uint point_index, Pixel tolerance)
{
    this->segments[segment_index].seed_points[point_index].tolerance = tolerance;
}
