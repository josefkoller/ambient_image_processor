#ifndef RawImage_H
#define RawImage_H


struct RawImage
{
    typedef unsigned int uint;
    typedef float Pixel;
    typedef Pixel* PixelPointer;
    typedef RawImage* Pointer;

    PixelPointer pixel_pointer;
public:
    struct Size {
        uint x;
        uint y;
    };
    typedef Size Index;

    const Size size;
    const uint pixel_count;

    RawImage(const Size size);
    RawImage(const RawImage& cloning_image);

    ~RawImage();

    void setPixel(uint x, uint y, Pixel pixel_value);

    void setPixel(uint i, Pixel pixel_value);

    Pixel getPixel(uint i);

    Pixel getPixel(uint x, uint y);

    uint oneDimensionalIndex(uint x, uint y);
};

#endif // RawImage_H
