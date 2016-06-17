#include "ConjugateGradientAlgorithm.h"

#include "cuda_host_helper.cuh"
#include "ImageVectorOperations.h"
#include "ImageMatrix.h"

ConjugateGradientAlgorithm::ConjugateGradientAlgorithm()
{
}

template<typename Pixel>
void ConjugateGradientAlgorithm::solveLinearEquationSystem(ImageMatrix<Pixel>* A,
                  Pixel* f, Pixel* x,
                  const Pixel epsilon)
{
    // init
    Dimension n = A->voxel_count;
    Pixel* r = cudaMalloc<Pixel>(n);
    Pixel* p = cudaMalloc<Pixel>(n);
    Pixel* temp = cudaMalloc<Pixel>(n);

    ImageVectorOperations::assign(r, p, n);

    Pixel rho = ImageVectorOperations::scalarProduct(r, r, temp, n);
    if(rho < epsilon)
    {
        cudaFree(r);
        cudaFree(p);
        cudaFree(temp);
        return;
    }
    ImageVectorOperations::matrixVectorMultiply(A, x, r);
    ImageVectorOperations::subtract(r, f, r, n);

    Pixel sigma = 0;
    Pixel alpha = 0;
    Pixel rho2 = 0;
    Pixel* s = cudaMalloc<Pixel>(n);
    // iterate
    for(int k = 0; k < n - 2; k++)
    {
        ImageVectorOperations::matrixVectorMultiply(A, p, s);
        sigma = ImageVectorOperations::scalarProduct(s, p, temp, n);
        alpha = rho / sigma;

        ImageVectorOperations::scale(p, alpha, temp, n);
        ImageVectorOperations::subtract(x, temp, x, n);

        ImageVectorOperations::scale(s, alpha, temp, n);
        ImageVectorOperations::subtract(r, temp, r, n);

        rho2 = ImageVectorOperations::scalarProduct(r, r, temp, n);
        if(rho2 < epsilon * rho)
        {
            cudaFree(r);
            cudaFree(p);
            cudaFree(temp);
            cudaFree(s);
            return;
        }

        ImageVectorOperations::scale(p, rho2/rho, temp, n);
        ImageVectorOperations::add(r, temp, p, n);
    }

    cudaFree(r);
    cudaFree(p);
    cudaFree(temp);
    cudaFree(s);
}

template void ConjugateGradientAlgorithm::solveLinearEquationSystem(ImageMatrix<double>* A,
                  double* f, double* x0,
                  const double epsilon);
template void ConjugateGradientAlgorithm::solveLinearEquationSystem(ImageMatrix<float>* A,
                  float* f, float* x0,
                  const float epsilon);

