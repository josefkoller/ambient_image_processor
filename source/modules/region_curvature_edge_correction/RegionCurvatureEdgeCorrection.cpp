#include "RegionCurvatureEdgeCorrection.h"
#include "ui_RegionCurvatureEdgeCorrection.h"

#include "RegionCurvatureEdgeCorrectionProcessor.h"

RegionCurvatureEdgeCorrection::RegionCurvatureEdgeCorrection(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::RegionCurvatureEdgeCorrection)
{
    ui->setupUi(this);
}

RegionCurvatureEdgeCorrection::~RegionCurvatureEdgeCorrection()
{
    delete ui;
}

void RegionCurvatureEdgeCorrection::on_set_seed_point_button_clicked()
{
    this->ui->set_seed_point_button->setFlat(
                ! this->ui->set_seed_point_button->isFlat());
}

void RegionCurvatureEdgeCorrection::registerModule(ImageWidget *image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(image_widget, &ImageWidget::mousePressedOnImage,
            this, &RegionCurvatureEdgeCorrection::mousePressedOnImage);
}

void RegionCurvatureEdgeCorrection::mousePressedOnImage(Qt::MouseButton button, ITKImage::Index position)
{
    if(!this->ui->set_seed_point_button->isFlat())
        return;

    this->setSeedPoint(position);
    this->ui->set_seed_point_button->setFlat(
                ! this->ui->set_seed_point_button->isFlat());
}

void RegionCurvatureEdgeCorrection::setSeedPoint(ITKImage::Index position)
{
    this->seed_point = position;
    this->ui->seed_point_label->setText(ITKImage::indexToText(position));
}

void RegionCurvatureEdgeCorrection::on_perform_button_clicked()
{
    this->processInWorkerThread();
}

ITKImage RegionCurvatureEdgeCorrection::processImage(ITKImage image)
{
    return RegionCurvatureEdgeCorrectionProcessor::process(
                image,
                this->seed_point, this->ui->tolerance_spinbox->value(),
                this->ui->count_of_pixels_to_leave_spinbox->value(),
                this->ui->count_of_node_pixels_spinbox->value(),
                this->ui->count_of_pixels_to_generate_spinbox->value()
                );
}
