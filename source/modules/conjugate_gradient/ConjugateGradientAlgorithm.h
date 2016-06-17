#ifndef CONJUGATEGRADIENTALGORITHM_H
#define CONJUGATEGRADIENTALGORITHM_H

template<typename Pixel>
class ImageMatrix;

class ConjugateGradientAlgorithm
{
private:
    typedef const unsigned int Dimension;
    ConjugateGradientAlgorithm();

public:
    template<typename Pixel>
    static void solveLinearEquationSystem(ImageMatrix<Pixel>* A,
                      Pixel* f, Pixel* x0,
                      const Pixel epsilon);

    template<typename Pixel>
    static void solvePoissonEquation(
            Pixel* f, Pixel* x0, Dimension image_width, Dimension image_height, Dimension image_depth,
            const Pixel epsilon);
};

#endif // CONJUGATEGRADIENTALGORITHM_H
