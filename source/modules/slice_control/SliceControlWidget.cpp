#include "SliceControlWidget.h"
#include "ui_SliceControlWidget.h"

SliceControlWidget::SliceControlWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::SliceControlWidget),
    image(ITKImage::Null),
    visible_slice_index(0)
{
    ui->setupUi(this);
}

SliceControlWidget::~SliceControlWidget()
{
    delete ui;
}

void SliceControlWidget::registerModule(ImageWidget *image_widget)
{
    connect(image_widget, &ImageWidget::imageChanged,
            this, [this](ITKImage& image){
        this->image = image;
        this->setInputRanges();

        if(this->image.getImageDimension() < 3) {
            auto container = dynamic_cast<QWidget*>(this->parent());
            container->setVisible(false);
        }
    });

    connect(this, &SliceControlWidget::sliceIndexChanged,
            image_widget, &ImageWidget::handleRepaintImage);
}

void SliceControlWidget::on_slice_spinbox_valueChanged(int user_slice_index)
{
    if(user_slice_index != this->visible_slice_index)
    {
        this->setSliceIndex(user_slice_index);
    }
}

void SliceControlWidget::on_slice_slider_valueChanged(int user_slice_index)
{
    if(user_slice_index != this->visible_slice_index)
    {
        this->setSliceIndex(user_slice_index);
    }
}

uint SliceControlWidget::userSliceIndex() const
{
    return this->ui->slice_slider->value();
}

void SliceControlWidget::setSliceIndex(uint slice_index)
{
    if(this->image.isNull())
        return;

    if(slice_index < 0 || slice_index >= this->image.getDepth())
    {
        std::cerr << "invalid slice_index for this image" << std::endl << std::flush;
        return;
    }

    this->visible_slice_index = slice_index;

    if(this->ui->slice_slider->value() != slice_index)
        this->ui->slice_slider->setValue(slice_index);

    if(this->ui->slice_spinbox->value() != slice_index)
        this->ui->slice_spinbox->setValue(slice_index);

    emit this->sliceIndexChanged(slice_index);
}

void SliceControlWidget::connectTo(BaseModule *other)
{
    auto other_module = dynamic_cast<SliceControlWidget*>(other);
    if(other_module == nullptr)
        return;

    connect(other_module, &SliceControlWidget::sliceIndexChanged,
            this, &SliceControlWidget::connectedSliceControlChanged);
}

void SliceControlWidget::connectedSliceControlChanged(uint slice_index)
{
    this->setSliceIndex(slice_index);
}

void SliceControlWidget::setInputRanges()
{
    if(this->image.isNull())
        return;

    this->ui->slice_slider->setMinimum(0); // first slice gets slice index 0
    this->ui->slice_slider->setMaximum(this->image.getDepth() - 1);

    this->ui->slice_spinbox->setMinimum(this->ui->slice_slider->minimum());
    this->ui->slice_spinbox->setMaximum(this->ui->slice_slider->maximum());
}

void SliceControlWidget::on_slice_slider_sliderMoved(int user_slice_index)
{

}