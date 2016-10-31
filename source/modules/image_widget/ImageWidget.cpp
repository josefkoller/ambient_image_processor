#include "ImageWidget.h"
#include "ui_ImageWidget.h"

#include <QPainter>
#include <QFileDialog>
#include <QDateTime>
#include <QMenuBar>

#include "ImageViewWidget.h"
#include "SliceControlWidget.h"
#include "LineProfileWidget.h"
#include "UnsharpMaskingWidget.h"
#include "MultiScaleRetinexWidget.h"
#include "NonLocalGradientWidget.h"
#include "RegionGrowingSegmentationWidget.h"
#include "HistogramWidget.h"
#include "ImageInformationWidget.h"
#include "ShrinkWidget.h"
#include "ThresholdFilterWidget.h"
#include "ExtractWidget.h"
#include "BilateralFilterWidget.h"
#include "ManualMultiplicativeDeshade.h"
#include "BinaryOperationsWidget.h"
#include "ConvolutionWidget.h"
#include "RescaleIntensityWidget.h"
#include "UnaryOperationsWidget.h"
#include "MorphologicalFilterWidget.h"
#include "ImageViewControlWidget.h"
#include "ConjugateGradientWidget.h"

#include "TGVWidget.h"
#include "TGVLambdasWidget.h"
#include "TGV3Widget.h"
#include "TGVKWidget.h"

#include "ResizeWidget.h"
#include "OriginSpacingWidget.h"

#include "TGVKDeshadeWidget.h"

/*
#include "TGVDeshadeWidget.h"
#include "TGV3DeshadeWidget.h"
#include "TGVDeshadeMaskedWidget.h"
*/

#include "TGVKDeshadeDownsampledWidget.h"
#include "TGVKDeshadeMaskedWidget.h"

ImageWidget::ImageWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ImageWidget),
    image(nullptr),
    output_widget(this),
    output_widget2(this)
{
    ui->setupUi(this);

    connect(this, SIGNAL(fireStatusTextChange(QString)),
            this, SLOT(handleStatusTextChange(QString)));

    qRegisterMetaType<ITKImage>("ITKImage");
    qRegisterMetaType<ITKImage>("ITKImage::Index");

    connect(this, &ImageWidget::fireImageChange,
            this, &ImageWidget::handleImageChange);


    this->ui->operations_panel->setVisible(false);

    // define modules
    auto module_parent = this->ui->operations_panel;

    auto region_growing_segmentation_widget =
            new RegionGrowingSegmentationWidget("Region Growing Segmentation", module_parent);
    auto non_local_gradient_widget = new NonLocalGradientWidget("Non-local Gradient", module_parent);
    auto tgv_widget = new TGVWidget("TGV Filter", module_parent);
    auto tgv_lambdas_widget = new TGVLambdasWidget("TGV Lambdas", module_parent);

    /*
    auto tgv_deshade_widget = new TGVDeshadeWidget("TGV2 Deshade", module_parent);
    auto tgv3_deshade_widget = new TGV3DeshadeWidget("TGV3 Deshade", module_parent);
    auto tgv_deshade_masked_widget = new TGVDeshadeMaskedWidget("TGV Deshade Masked", module_parent);
    */


    auto tgv3_widget = new TGV3Widget("TGV3 Filter", module_parent);
    auto tgvk_widget = new TGVKWidget("TGVk Filter", module_parent);

    auto tgvk_deshade_widget = new TGVKDeshadeWidget("TGV-DCT Deshade", module_parent);
    auto tgvk_deshade_masked_widget = new TGVKDeshadeMaskedWidget("TGV-DCT Deshade Masked", module_parent);
    auto tgvk_deshade_downsampled_widget = new TGVKDeshadeDownsampledWidget("TGV-DCT Deshade Downsampled", module_parent);


    this->image_view_widget = new ImageViewWidget("Image View", this->ui->image_frame);
    this->slice_control_widget = new SliceControlWidget("Slice Control", this->ui->slice_control_widget_frame);

    auto image_view_control_widget = new ImageViewControlWidget("Image View Control", module_parent);

    modules.push_back(this->image_view_widget);
    modules.push_back(this->slice_control_widget);
    modules.push_back(new ImageInformationWidget("Image Information", module_parent));
    modules.push_back(new OriginSpacingWidget("Origin Spacing", module_parent));
    modules.push_back(image_view_control_widget);
    modules.push_back(new LineProfileWidget("Line Profile", module_parent));
    modules.push_back(new HistogramWidget("Histogram", module_parent));
    modules.push_back(new BinaryOperationsWidget("Binary Operations", module_parent));
    modules.push_back(new UnaryOperationsWidget("Unary Operations", module_parent));
    modules.push_back(new ThresholdFilterWidget("Threshold", module_parent));
    modules.push_back(new MorphologicalFilterWidget("Morphological Filter", module_parent));
    modules.push_back(new ConvolutionWidget("3x3x3 Convolution", module_parent));
    modules.push_back(new RescaleIntensityWidget("Rescale Intensity", module_parent));
    modules.push_back(new ShrinkWidget("Shrink", module_parent));
    modules.push_back(new ExtractWidget("Extract", module_parent));
    modules.push_back(new ResizeWidget("Resize", module_parent));
    modules.push_back(non_local_gradient_widget);
    modules.push_back(region_growing_segmentation_widget);
    modules.push_back(new BilateralFilterWidget("Bilateral Filter", module_parent));
    modules.push_back(tgv_widget);
    modules.push_back(tgv3_widget);
    modules.push_back(tgvk_widget);
    modules.push_back(tgv_lambdas_widget);

    modules.push_back(new UnsharpMaskingWidget("Unsharp Masking", module_parent));
    modules.push_back(new MultiScaleRetinexWidget("Multiscale Retinex", module_parent));

    modules.push_back(new ManualMultiplicativeDeshade("Manual Multiplicative Deshade", module_parent));

    /*
    modules.push_back(tgv_deshade_widget);
    modules.push_back(tgv3_deshade_widget);
    modules.push_back(tgv_deshade_masked_widget);
    */

    modules.push_back(tgvk_deshade_widget);
    modules.push_back(tgvk_deshade_masked_widget);
    modules.push_back(tgvk_deshade_downsampled_widget);

    modules.push_back(new ConjugateGradientWidget("Conjugate Gradient", module_parent));

    // register modules and add widget modules
    module_parent->hide();
    module_parent->setUpdatesEnabled(false);
    uint index = 0;
    for(auto module : modules)
    {
        auto widget = dynamic_cast<BaseModuleWidget*>(module);
        if(widget != nullptr &&
                widget != slice_control_widget &&
                widget != image_view_widget)
            module_parent->insertTab(index++, widget, module->getTitle());

        std::cout << "registering module: " << module->getTitle().toStdString() << std::endl;
        module->registerModule(this);
    }

    this->ui->image_frame->layout()->addWidget(image_view_widget);
    this->ui->slice_control_widget_frame->layout()->addWidget(slice_control_widget);

    module_parent->setUpdatesEnabled(true);
    module_parent->show();

    // create menu entry for widget modules
    QMenuBar* menu_bar = new QMenuBar();
    this->image_menu = new QMenu("Image");
    QAction* load_action = image_menu->addAction("Load File");
    this->connect(load_action, &QAction::triggered, this, [this]() {
        this->on_load_button_clicked();
    });
    QAction* save_action = image_menu->addAction("Save File");
    this->connect(save_action, &QAction::triggered, this, [this]() {
        this->on_save_button_clicked();
    });
    image_menu->addSeparator();
    QAction* save_with_overlays_action = image_menu->addAction("Save File with Overlays");
    this->connect(save_with_overlays_action, &QAction::triggered, this, [this]() {
        this->image_view_widget->save_file_with_overlays();
    });

    auto line_profile_module = this->getModuleByName("Line Profile");
    if(line_profile_module)
    {
        QAction* save_line_profile_action = image_menu->addAction("Save Line Profile");
        this->connect(save_line_profile_action, &QAction::triggered, this, [line_profile_module]() {
            auto line_profile_module_casted = dynamic_cast<LineProfileWidget*>(line_profile_module);
            line_profile_module_casted->save_to_file();
        });
    }
    image_menu->addSeparator();
    QAction* load_hsv = image_menu->addAction("Load Color File HSV");
    this->connect(load_hsv, &QAction::triggered, this, [this]() {
        this->load_hsv_clicked();
    });
    QAction* save_hsv = image_menu->addAction("Save into Color File HSV");
    this->connect(save_hsv, &QAction::triggered, this, [this]() {
        this->save_hsv_clicked();
    });
    image_menu->addSeparator();
    QAction* load_color_to_view_only = image_menu->addAction("Load Color File to View only");
    this->connect(load_color_to_view_only, &QAction::triggered, this, [this]() {
        this->image_view_widget->load_color_to_view_only_clicked();
    });

    menu_bar->addMenu(image_menu);
    QMenu *tools_menu = new QMenu("Tools");
    for(auto module : modules)
    {
        auto widget = dynamic_cast<BaseModuleWidget*>(module);

        if(widget == nullptr ||
           widget == slice_control_widget ||
           widget == image_view_widget)
            continue;

        QAction* module_action = tools_menu->addAction(module->getTitle());

        if(widget->getTitle() == "Histogram" ||
           widget->getTitle() == "Rescale Intensity" ||
           widget->getTitle() == "Resize" ||
           widget->getTitle() == "Bilateral Filter" ||
           widget->getTitle() == "TGV Lambdas" ||
           widget->getTitle() == "Multiscale Retinex" ||
           widget->getTitle() == "Manual Multiplicative Deshade" ||
           widget->getTitle() == "TGV-DCT Deshade Downsampled")
            tools_menu->addSeparator();

        this->connect(module_action, &QAction::triggered, this, [this, widget]() {
            this->ui->operations_panel->setCurrentWidget(widget);
        });
    }
    menu_bar->addMenu(tools_menu);
    this->layout()->setMenuBar(menu_bar);

    // connect modules...

    connect(image_view_control_widget, &ImageViewControlWidget::doRescaleChanged,
            this->image_view_widget, &ImageViewWidget::doRescaleChanged);
    connect(image_view_control_widget, &ImageViewControlWidget::doMultiplyChanged,
            this->image_view_widget, &ImageViewWidget::doMultiplyChanged);
    connect(image_view_control_widget, &ImageViewControlWidget::useWindowChanged,
            this->image_view_widget, &ImageViewWidget::useWindowChanged);

    // iteration finished callback...
    auto iteration_finished_callback = [this](uint index, uint count, ITKImage u) {
        emit this->fireStatusTextChange(QString("iteration %1 / %2").arg(
                                            QString::number(index+1),
                                            QString::number(count)));
        emit this->output_widget->fireImageChange(u.getPointer());
        return false;
    };
    tgv_widget->setIterationFinishedCallback(iteration_finished_callback);
    tgv_lambdas_widget->setIterationFinishedCallback(iteration_finished_callback);
    tgv3_widget->setIterationFinishedCallback(iteration_finished_callback);
    tgvk_widget->setIterationFinishedCallback(iteration_finished_callback);

    /*
    tgv_deshade_masked_widget->setIterationFinishedCallback(iteration_finished_callback);
    tgv_deshade_widget->setIterationFinishedCallback(iteration_finished_callback);
    tgv3_deshade_widget->setIterationFinishedCallback(iteration_finished_callback);
    */

    tgvk_deshade_widget->setIterationFinishedCallback(iteration_finished_callback);
    tgvk_deshade_downsampled_widget->setIterationFinishedCallback(iteration_finished_callback);
    tgvk_deshade_masked_widget->setIterationFinishedCallback(iteration_finished_callback);
}

ImageWidget::~ImageWidget()
{
    delete ui;
}


void ImageWidget::setImage(ITKImage image)
{
    this->image = image.clone();

    this->setMinimumSizeToImage();

    emit this->imageChanged(this->image);
}

BaseModule* ImageWidget::getModuleByName(QString module_title) const
{
    for(auto module : modules)
        if(module->getTitle() == module_title)
            return module;
    return nullptr;

}
void ImageWidget::connectModule(QString module_title, ImageWidget* other_image_widget)
{
    auto module1 = this->getModuleByName(module_title);
    auto module2 = other_image_widget->getModuleByName(module_title);

    if(module1 == nullptr || module2 == nullptr)
        return;

    module1->connectTo(module2);
}

void ImageWidget::on_load_button_clicked()
{
    QString file_name = QFileDialog::getOpenFileName(this, "open volume file");
    if(file_name == QString::null || !QFile(file_name).exists())
        return;

    this->setImage(ITKImage::read(file_name.toStdString()));
}

void ImageWidget::on_save_button_clicked()
{
    if(this->image.isNull())
        return;

    QString file_name = QFileDialog::getSaveFileName(this, "save volume file");
    if(file_name.isNull())
        return;

    this->image.write(file_name.toStdString());
}

void ImageWidget::save_hsv_clicked()
{
    if(this->image.isNull())
        return;

    QString template_file_name = QFileDialog::getOpenFileName(this, "open template color file for the H and S channel");
    if(template_file_name == QString::null || !QFile(template_file_name).exists())
        return;

    QString file_name = QFileDialog::getSaveFileName(this, "save V channel into color file");
    if(file_name.isNull())
        return;

    QFile::copy(template_file_name, file_name);

    this->image.write_hsv(file_name.toStdString());
}

void ImageWidget::load_hsv_clicked()
{
    QString file_name = QFileDialog::getOpenFileName(this, "open volume file");
    if(file_name == QString::null || !QFile(file_name).exists())
        return;


    this->setImage(ITKImage::read_hsv(file_name.toStdString()));
}

void ImageWidget::setMinimumSizeToImage()
{
    if(this->image.isNull())
        return;

    const int border = 50;
    this->ui->image_frame->setMinimumSize(
                this->image.width + border*2,
                this->image.height + border*2);
}

void ImageWidget::setOutputWidget(ImageWidget* output_widget)
{
    this->output_widget = output_widget;

    output_widget->image_menu->addSeparator();
    auto action = output_widget->image_menu->addAction("Swap Input and Output");
    this->connect(action, &QAction::triggered, this, [this, output_widget]() {
        auto source_image = this->getImage();
        this->setImage(this->output_widget->getImage());
        output_widget->setImage(source_image);
    });
    this->image_menu->addSeparator();
    this->image_menu->addAction(action);
}

ImageWidget* ImageWidget::getOutputWidget() const
{
    return this->output_widget;
}

void ImageWidget::setOutputWidget2(ImageWidget* output_widget)
{
    this->output_widget2 = output_widget;
}

void ImageWidget::setOutputWidget3(ImageWidget* output_widget)
{
    this->output_widget3 = output_widget;
}


void ImageWidget::handleStatusTextChange(QString text)
{
    this->ui->status_bar->setText(text);
    this->ui->status_bar->repaint();
}

void ImageWidget::handleImageChange(ITKImage image)
{
    this->setImage(image);
}

void ImageWidget::setPage(unsigned char page_index)
{
    this->ui->operations_panel->setCurrentIndex(page_index);
}
