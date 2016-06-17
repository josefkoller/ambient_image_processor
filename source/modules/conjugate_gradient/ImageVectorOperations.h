#ifndef IMAGEVECTOROPERATIONS_H
#define IMAGEVECTOROPERATIONS_H

template<typename Pixel>
class ImageMatrix;

class ImageVectorOperations
{
private:
    ImageVectorOperations();

    typedef const unsigned int Dimension;
public:
    template<typename Pixel>
    static void setZeros(Pixel* vector, Dimension voxel_count);

    template<typename Pixel>
    static Pixel scalarProduct(Pixel* image1, Pixel* image2, Pixel* temp, Dimension voxel_count);
    template<typename Pixel>
    static void add(Pixel* image1, Pixel* image2, Pixel* result, Dimension voxel_count);
    template<typename Pixel>
    static void subtract(Pixel* image1, Pixel* image2, Pixel* result, Dimension voxel_count);
    template<typename Pixel>
    static void scale(Pixel* image1, Pixel factor, Pixel* result, Dimension voxel_count);
    template<typename Pixel>
    static void assign(Pixel* source, Pixel* destination, Dimension voxel_count);
    template<typename Pixel>
    static void matrixVectorMultiply(ImageMatrix<Pixel>* matrix, Pixel* vector, Pixel* result);


    template<typename Pixel>
    static void laplace(Pixel* vector, Pixel* result,
                        Dimension image_width, Dimension image_height, Dimension image_depth);
};

#endif // IMAGEVECTOROPERATIONS_H
