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

    PixelIndex left() { return PixelIndex(x-1, y, z); }
    PixelIndex leftTop() { return PixelIndex(x-1, y-1, z); }
    PixelIndex leftBottom() { return PixelIndex(x-1, y+1, z); }
    PixelIndex right() { return PixelIndex(x+1, y, z); }
    PixelIndex rightTop() { return PixelIndex(x+1, y-1, z); }
    PixelIndex rightBottom() { return PixelIndex(x+1, y+1, z); }
    PixelIndex top() { return PixelIndex(x, y-1, z); }
    PixelIndex bottom() { return PixelIndex(x, y+1, z); }

    PixelIndex front() { return PixelIndex(x, y, z-1); }
    PixelIndex back() { return PixelIndex(x, y, z+1); }

    PixelIndex frontLeft() { return PixelIndex(x-1, y, z-1); }
    PixelIndex frontLeftTop() { return PixelIndex(x-1, y-1, z-1); }
    PixelIndex frontLeftBottom() { return PixelIndex(x-1, y+1, z-1); }
    PixelIndex frontRight() { return PixelIndex(x+1, y, z-1); }
    PixelIndex frontRightTop() { return PixelIndex(x+1, y-1, z-1); }
    PixelIndex frontRightBottom() { return PixelIndex(x+1, y+1, z-1); }
    PixelIndex frontTop() { return PixelIndex(x, y-1, z); }
    PixelIndex frontBottom() { return PixelIndex(x, y+1, z-1); }

    PixelIndex backLeft() { return PixelIndex(x-1, y, z+1); }
    PixelIndex backLeftTop() { return PixelIndex(x-1, y-1, z+1); }
    PixelIndex backLeftBottom() { return PixelIndex(x-1, y+1, z+1); }
    PixelIndex backRight() { return PixelIndex(x+1, y, z+1); }
    PixelIndex backRightTop() { return PixelIndex(x+1, y-1, z+1); }
    PixelIndex backRightBottom() { return PixelIndex(x+1, y+1, z+1); }
    PixelIndex backTop() { return PixelIndex(x, y-1, z+1); }
    PixelIndex backBottom() { return PixelIndex(x, y+1, z+1); }

    std::vector<PixelIndex> collectNeighbours(PixelIndex size);

    bool isInside(PixelIndex size);
};

#endif // PIXELINDEX_H

