#include "UnaryOperationsWidget.h"
#include "ui_UnaryOperationsWidget.h"

#include "CudaImageOperationsProcessor.h"
#include <itkRescaleIntensityImageFilter.h>

UnaryOperationsWidget::UnaryOperationsWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::UnaryOperationsWidget)
{
    ui->setupUi(this);
}

UnaryOperationsWidget::~UnaryOperationsWidget()
{
    delete ui;
}

ITKImage UnaryOperationsWidget::processImage(ITKImage image)
{
    if(this->ui->invert_checkbox->isChecked())
        return CudaImageOperationsProcessor::invert(image);

    if(this->ui->binarize_checkbox->isChecked())
        return CudaImageOperationsProcessor::binarize(image);

    if(this->ui->dct_checkbox->isChecked())
        return CudaImageOperationsProcessor::cosineTransform(image);
    if(this->ui->idct_checkbox->isChecked())
        return CudaImageOperationsProcessor::inverseCosineTransform(image);

    if(this->ui->exp_checkbox->isChecked())
    {
        image = CudaImageOperationsProcessor::exp(image);
        auto one = image.clone();
        one.setEachPixel([] (uint,uint,uint) { return 1.0; });
        return CudaImageOperationsProcessor::subtract(image, one);
    }

    if(this->ui->log_checkbox->isChecked())
    {
        image = this->rescale(image);
        return CudaImageOperationsProcessor::log(image);
    }
}

ITKImage UnaryOperationsWidget::rescale(ITKImage image)
{
    typedef ITKImage::InnerITKImage Image;
    typedef itk::RescaleIntensityImageFilter<Image, Image> RescaleFilter;
    RescaleFilter::Pointer rescale_filter = RescaleFilter::New();
    rescale_filter->SetInput(image.getPointer());
    rescale_filter->SetOutputMinimum(1);
    rescale_filter->SetOutputMaximum(2);
    rescale_filter->Update();

    Image::Pointer result = rescale_filter->GetOutput();
    result->DisconnectPipeline();

    return ITKImage(result);
}

void UnaryOperationsWidget::on_perform_button_clicked()
{
    this->processInWorkerThread();
}
