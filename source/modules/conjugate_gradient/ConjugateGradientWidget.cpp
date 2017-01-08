/*
    Ambient Image Processor - A tool to perform several imaging tasks
    
    Copyright (C) 2016 Josef Koller

    https://github.com/josefkoller/ambient_image_processor    
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "ConjugateGradientWidget.h"
#include "ui_ConjugateGradientWidget.h"

#include "ImageMatrixGradientFactory.h"
#include "ConjugateGradientAlgorithm.h"
#include "ImageVectorOperations.h"

#include "cuda_host_helper.cuh"

#include "CudaImageOperationsProcessor.h"

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

ITKImage ConjugateGradientWidget::processImage(ITKImage image)
{

    if(!this->ui->conjugate_gradient_checkbox->isChecked())
    {
        auto image2 = CudaImageOperationsProcessor::cosineTransform(image);
        auto image3 = CudaImageOperationsProcessor::solvePoissonInCosineDomain(image2);
        return CudaImageOperationsProcessor::inverseCosineTransform(image3);
    }

    typedef ITKImage::PixelType Pixel;

    Pixel* image_pixels = image.cloneToCudaPixelArray();
    Pixel* result_pixels = cudaMalloc<Pixel>(image.voxel_count);
    Pixel epsilon = this->ui->epsilon_spinbox->value();

    ImageVectorOperations::setZeros(result_pixels, image.voxel_count); // initial guess

    /*
    auto laplace_operator = ImageMatrixGradientFactory::laplace<Pixel>(
                image.width, image.height, image.depth);
    ConjugateGradientAlgorithm::solveLinearEquationSystem(laplace_operator, image_pixels, result_pixels, epsilon);
    delete laplace_operator;
    */

    ConjugateGradientAlgorithm::solvePoissonEquation(image_pixels, result_pixels,
       image.width, image.height, image.depth, epsilon);

    cudaFree(image_pixels);

    auto result = ITKImage(image.width, image.height, image.depth, result_pixels);
    cudaFree(result_pixels);
    return result;
}

void ConjugateGradientWidget::on_solve_poisson_button_clicked()
{
    this->processInWorkerThread();
}
