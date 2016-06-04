#ifndef PIXELINDEX_H
#define PIXELINDEX_H

#include <vector>

struct PixelIndex
{
    typedef unsigned int uint;

    uint x,y,z;
    PixelIndex() : x(0), y(0), z(0) {}
    PixelIndex(uint x, uint y, uint z) : x(x), y(y), z(z) {}
    PixelIndex(const PixelIndex& clone) : x(clone.x), y(clone.y), z(clone.z) {}
    PixelIndex(uint linear_index, PixelIndex size);

    PixelIndex left() const { return PixelIndex(x-1, y, z); }
    PixelIndex leftTop() const { return PixelIndex(x-1, y-1, z); }
    PixelIndex leftBottom() const { return PixelIndex(x-1, y+1, z); }
    PixelIndex right() const { return PixelIndex(x+1, y, z); }
    PixelIndex rightTop() const { return PixelIndex(x+1, y-1, z); }
    PixelIndex rightBottom() const { return PixelIndex(x+1, y+1, z); }
    PixelIndex top() const { return PixelIndex(x, y-1, z); }
    PixelIndex bottom() const { return PixelIndex(x, y+1, z); }

    PixelIndex front() const { return PixelIndex(x, y, z-1); }
    PixelIndex back() const { return PixelIndex(x, y, z+1); }

    PixelIndex frontLeft() const { return PixelIndex(x-1, y, z-1); }
    PixelIndex frontLeftTop() const { return PixelIndex(x-1, y-1, z-1); }
    PixelIndex frontLeftBottom() const { return PixelIndex(x-1, y+1, z-1); }
    PixelIndex frontRight() const { return PixelIndex(x+1, y, z-1); }
    PixelIndex frontRightTop() const { return PixelIndex(x+1, y-1, z-1); }
    PixelIndex frontRightBottom() const { return PixelIndex(x+1, y+1, z-1); }
    PixelIndex frontTop() const { return PixelIndex(x, y-1, z); }
    PixelIndex frontBottom() const { return PixelIndex(x, y+1, z-1); }

    PixelIndex backLeft() const { return PixelIndex(x-1, y, z+1); }
    PixelIndex backLeftTop() const { return PixelIndex(x-1, y-1, z+1); }
    PixelIndex backLeftBottom() const { return PixelIndex(x-1, y+1, z+1); }
    PixelIndex backRight() const { return PixelIndex(x+1, y, z+1); }
    PixelIndex backRightTop() const { return PixelIndex(x+1, y-1, z+1); }
    PixelIndex backRightBottom() const { return PixelIndex(x+1, y+1, z+1); }
    PixelIndex backTop() const { return PixelIndex(x, y-1, z+1); }
    PixelIndex backBottom() const { return PixelIndex(x, y+1, z+1); }

    std::vector<PixelIndex> collectNeighbours(PixelIndex size) const;
    bool isInside(PixelIndex size) const;
    uint linearIndex(PixelIndex size) const;
};

#endif // PIXELINDEX_H

