#include "ManualMultiplicativeDeshade.h"
#include "ui_ManualMultiplicativeDeshade.h"

#include "CudaImageOperationsProcessor.h"

ManualMultiplicativeDeshade::ManualMultiplicativeDeshade(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::ManualMultiplicativeDeshade)
{
    ui->setupUi(this);

    this->generateKernel();
    this->initShading();
}

ManualMultiplicativeDeshade::~ManualMultiplicativeDeshade()
{
    delete ui;
}

void ManualMultiplicativeDeshade::initShading() {
    auto source_image = this->getSourceImage();

    this->shading = ITKImage(source_image.width, source_image.height, source_image.depth);
    this->shading.setEachPixel([](uint,uint,uint) {
        return 1.0;
    });
}

void ManualMultiplicativeDeshade::registerModule(ImageWidget *image_widget)
{
    BaseModuleWidget::registerModule(image_widget);


    connect(image_widget, &ImageWidget::mousePressedOnImage,
            this, &ManualMultiplicativeDeshade::mousePressedOnImage);

}


void ManualMultiplicativeDeshade::mousePressedOnImage(Qt::MouseButton button, ITKImage::Index position)
{
    if(!this->ui->edit_kernel_checkbox->isChecked())
        return;

    this->increase = button == Qt::LeftButton;
    this->cursor_position = position;

    this->processInWorkerThread();
}

void ManualMultiplicativeDeshade::generateKernel()
{
    auto kernel_sigma = this->ui->kernel_sigma_spinbox->value();
    auto kernel_size = this->ui->kernel_size_spinbox->value();

    if(this->getSourceImage().depth > 1)
        this->kernel = ITKImage(kernel_size,kernel_size,kernel_size);
    else
        this->kernel = ITKImage(kernel_size,kernel_size,1);

    uint kernel_center_x = std::floor(kernel.width / 2.0f);
    uint kernel_center_y = std::floor(kernel.height / 2.0f);
    uint kernel_center_z = std::floor(kernel.depth / 2.0f);

    kernel.setEachPixel([kernel_center_x, kernel_center_y,kernel_center_z,kernel_sigma] (uint x, uint y, uint z) {
        uint xr = x - kernel_center_x;
        uint yr = y - kernel_center_y;
        uint zr = z - kernel_center_z;

        ITKImage::PixelType radius = std::sqrt(xr*xr + yr*yr + zr*zr);
        ITKImage::PixelType value = std::exp(-radius*radius / kernel_sigma);

        return value;
    });

    const ITKImage::PixelType offset = (this->ui->kernel_maximum_spinbox->value() - 1);

    kernel.foreachPixel([this, offset] (uint x, uint y, uint z, ITKImage::PixelType value) {
        this->kernel.setPixel(x,y,z, value * offset + 1); // [0..1] to [1..maximum]
    });
}

ITKImage ManualMultiplicativeDeshade::processImage(ITKImage image)
{
    if(!this->ui->edit_kernel_checkbox->isChecked())
        return ITKImage();

    if(this->shading.width == 0)
        this->initShading();

    uint kernel_center_x = std::floor(kernel.width / 2.0f);
    uint kernel_center_y = std::floor(kernel.height / 2.0f);
    uint kernel_center_z = std::floor(kernel.depth / 2.0f);

    uint x = this->cursor_position[0];
    uint y = this->cursor_position[1];
    uint z = this->cursor_position[2];

    for(int kz = 0; kz < kernel.depth; kz++)
    {
        for(int ky = 0; ky < kernel.height; ky++)
        {
            for(int kx = 0; kx < kernel.width; kx++)
            {
                const int kxa = x + (kx - kernel_center_x);
                const int kya = y + (ky - kernel_center_y);
                const int kza = z + (kz - kernel_center_z);

                if(kxa < 0 || kxa >= shading.width ||
                   kya < 0 || kya >= shading.height ||
                    kza < 0 || kza >= shading.depth)
                    continue;

                ITKImage::PixelType value = this->shading.getPixel(kxa, kya, kza);
                ITKImage::PixelType kernel_value = kernel.getPixel(kx,ky,kz);
                if(!this->increase)
                    kernel_value = 1.0 / kernel_value;

                value *= kernel_value;

                this->shading.setPixel(kxa, kya, kza, value);
            }
        }
    }
   // return shading;

    auto source_image = this->getSourceImage();
    return CudaImageOperationsProcessor::multiply(source_image, shading);
}

void ManualMultiplicativeDeshade::on_kernel_sigma_spinbox_editingFinished()
{
    this->generateKernel();
}

void ManualMultiplicativeDeshade::on_kernel_size_spinbox_editingFinished()
{
    this->generateKernel();
}

void ManualMultiplicativeDeshade::on_kernel_maximum_spinbox_editingFinished()
{
    this->generateKernel();
}

void ManualMultiplicativeDeshade::on_reset_shading_button_clicked()
{
    this->initShading();
}
