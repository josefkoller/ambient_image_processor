#include "OriginSpacingWidget.h"
#include "ui_OriginSpacingWidget.h"

OriginSpacingWidget::OriginSpacingWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::OriginSpacingWidget)
{
    ui->setupUi(this);
}

OriginSpacingWidget::~OriginSpacingWidget()
{
    delete ui;
}

void OriginSpacingWidget::registerModule(ImageWidget* image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(image_widget, &ImageWidget::imageChanged,
            [this](ITKImage image) {
        if(image.isNull())
            return;

        this->ui->originXSpinbox->setValue(image.getPointer()->GetOrigin()[0]);
        this->ui->originYSpinbox->setValue(image.getPointer()->GetOrigin()[1]);
        this->ui->spacingXSpinbox->setValue(image.getPointer()->GetSpacing()[0]);
        this->ui->spacingYSpinbox->setValue(image.getPointer()->GetSpacing()[1]);

        if(ITKImage::ImageDimension > 2) {
            this->ui->originZSpinbox->setValue(image.getPointer()->GetOrigin()[2]);
            this->ui->spacingZSpinbox->setValue(image.getPointer()->GetSpacing()[2]);
        }
    });
}

void OriginSpacingWidget::on_performButton_clicked()
{
    this->processInWorkerThread();
}

ITKImage OriginSpacingWidget::processImage(ITKImage image)
{
    ITKImage::InnerITKImage::SpacingType spacing;
    spacing[0] = this->ui->spacingXSpinbox->value();
    spacing[1] = this->ui->spacingYSpinbox->value();
    ITKImage::InnerITKImage::PointType origin;
    origin[0] = this->ui->originXSpinbox->value();
    origin[1] = this->ui->originYSpinbox->value();

    if(ITKImage::ImageDimension > 2) {
        spacing[2] = this->ui->spacingZSpinbox->value();
        origin[2] = this->ui->originZSpinbox->value();
    }

    auto result = image.clone();
    result.getPointer()->SetOrigin(origin);
    result.getPointer()->SetSpacing(spacing);
    return result;
}
