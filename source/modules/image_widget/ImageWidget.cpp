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
#include "SplineInterpolationWidget.h"
#include "ThresholdFilterWidget.h"
#include "ExtractWidget.h"
#include "BilateralFilterWidget.h"
#include "DeshadeSegmentedWidget.h"
#include "TGVWidget.h"
#include "CrosshairModule.h"
#include "TGVL1ThresholdGradientWidget.h"
#include "ManualMultiplicativeDeshade.h"
#include "TGVLambdasWidget.h"
#include "BinaryOperationsWidget.h"
#include "ConvolutionWidget.h"

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
    auto deshade_segmented_widget = new DeshadeSegmentedWidget("Deshade Segmented", module_parent);
    auto tgv_widget = new TGVWidget("TGV Filter", module_parent);
    auto tgv_lambdas_widget = new TGVLambdasWidget("TGV Lambdas", module_parent);

    this->image_view_widget = new ImageViewWidget("Image View", this->ui->image_frame);
    this->slice_control_widget = new SliceControlWidget("Slice Control", this->ui->slice_control_widget_frame);

    modules.push_back(this->image_view_widget);
    modules.push_back(this->slice_control_widget);
    modules.push_back(new ImageInformationWidget("Image Information", module_parent));
    modules.push_back(new HistogramWidget("Histogram", module_parent));
    modules.push_back(new ThresholdFilterWidget("Threshold", module_parent));
    modules.push_back(new LineProfileWidget("Line Profile", module_parent));
    modules.push_back(new ShrinkWidget("Shrink", module_parent));
    modules.push_back(new ExtractWidget("Extract", module_parent));
    modules.push_back(new UnsharpMaskingWidget("Unsharp Masking", module_parent));
    modules.push_back(new MultiScaleRetinexWidget("Multiscale Retinex", module_parent));
    modules.push_back(non_local_gradient_widget);
    modules.push_back(region_growing_segmentation_widget);
    modules.push_back(deshade_segmented_widget);
    modules.push_back(new SplineInterpolationWidget("Spline Interpolation", module_parent));
    modules.push_back(new BilateralFilterWidget("Bilateral Filter", module_parent));
    modules.push_back(tgv_widget);
    modules.push_back(new CrosshairModule("Bilateral Filter"));
    modules.push_back(new TGVL1ThresholdGradientWidget("TGVL1 Thresholded Gradient", module_parent));
    modules.push_back(new ManualMultiplicativeDeshade("Manual Multiplicative Deshade", module_parent));
    modules.push_back(tgv_lambdas_widget);
    modules.push_back(new BinaryOperationsWidget("Binary Operations", module_parent));
    modules.push_back(new ConvolutionWidget("3x3 Convolution", module_parent));


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
    QMenu *file_menu = new QMenu("File");
    QAction* load_action = file_menu->addAction("Load");
    this->connect(load_action, &QAction::triggered, this, [this]() {
        this->on_load_button_clicked();
    });
    QAction* save_action = file_menu->addAction("Save");
    this->connect(save_action, &QAction::triggered, this, [this]() {
        this->on_save_button_clicked();
    });
    menu_bar->addMenu(file_menu);
    QMenu *tools_menu = new QMenu("Tools");
    for(auto module : modules)
    {
        auto widget = dynamic_cast<BaseModuleWidget*>(module);
        if(widget == nullptr && widget != slice_control_widget)
            continue;

        QAction* module_action = tools_menu->addAction(module->getTitle());
        this->connect(module_action, &QAction::triggered, this, [this, widget]() {
            this->ui->operations_panel->setCurrentWidget(widget);
        });
    }
    menu_bar->addMenu(tools_menu);
    this->layout()->setMenuBar(menu_bar);

    // connect modules...
    region_growing_segmentation_widget->setKernelSigmaFetcher([non_local_gradient_widget]() {
        return non_local_gradient_widget->getKernelSigma();
    });
    region_growing_segmentation_widget->setKernelSizeFetcher([non_local_gradient_widget]() {
        return non_local_gradient_widget->getKernelSize();
    });

    deshade_segmented_widget->setSegmentsFetcher([region_growing_segmentation_widget]() {
        return region_growing_segmentation_widget->getSegments();
    });
    deshade_segmented_widget->setLabelImageFetcher([region_growing_segmentation_widget]() {
        return region_growing_segmentation_widget->getLabelImage();
    });

    auto iteration_finished_callback = [this](uint index, uint count, ITKImage u) {
        emit this->fireStatusTextChange(QString("iteration %1 / %2").arg(
                                            QString::number(index+1),
                                            QString::number(count)));
        emit this->output_widget->fireImageChange(u.getPointer());
        return false;
    };
    tgv_widget->setIterationFinishedCallback(iteration_finished_callback);
    tgv_lambdas_widget->setIterationFinishedCallback(iteration_finished_callback);
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
