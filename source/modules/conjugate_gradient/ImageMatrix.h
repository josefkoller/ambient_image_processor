#ifndef IMAGEMATRIX_H
#define IMAGEMATRIX_H

#include <cuda_runtime.h>

template<typename Pixel>
class ImageMatrix
{
private:
    typedef const unsigned int Dimension;

    Dimension element_count;

    dim3 block_dimension;
    dim3 grid_dimension;

public:
    Pixel* elements;
    Dimension voxel_count;

    ImageMatrix(Dimension voxel_count);
    ~ImageMatrix();

    void setZeros();

    void setPixelTransformation(Dimension source_pixel_index,
                                Dimension target_pixel_index,
                                Pixel factor);

    void transposed(ImageMatrix<Pixel>* transposed_matrix);
    void add(ImageMatrix<Pixel>* matrix2, ImageMatrix<Pixel>* result);
    void multiply(ImageMatrix<Pixel>* matrix2, ImageMatrix<Pixel>* result);
};

#endif // IMAGEMATRIX_H
