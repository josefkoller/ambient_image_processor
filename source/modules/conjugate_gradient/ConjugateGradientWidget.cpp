#include "ConjugateGradientWidget.h"
#include "ui_ConjugateGradientWidget.h"

#include "ImageMatrixGradientFactory.h"
#include "ConjugateGradientAlgorithm.h"

#include "cuda_host_helper.cuh"

ConjugateGradientWidget::ConjugateGradientWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::ConjugateGradientWidget)
{
    ui->setupUi(this);
}

ConjugateGradientWidget::~ConjugateGradientWidget()
{
    delete ui;
}

void ConjugateGradientWidget::on_pushButton_clicked()
{
    this->processInWorkerThread();
}

ITKImage ConjugateGradientWidget::processImage(ITKImage image)
{
    typedef ITKImage::PixelType Pixel;

    Pixel* image_pixels = image.cloneToCudaPixelArray();
    Pixel* result_pixels = cudaMalloc<Pixel>(image.voxel_count);

    auto laplace_operator = ImageMatrixGradientFactory::laplace<Pixel>(
                image.width, image.height, image.depth);

    Pixel epsilon = this->ui->epsilon_spinbox->value();

    ConjugateGradientAlgorithm::solveLinearEquationSystem(laplace_operator, image_pixels, result_pixels, epsilon);

    delete laplace_operator;
    cudaFree(image_pixels);

    auto result = ITKImage(image.width, image.height, image.depth, result_pixels);
    cudaFree(result_pixels);
    return result;
}
