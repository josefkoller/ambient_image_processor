#include "ImageWidget.h"
#include "ui_ImageWidget.h"

#include <QPainter>
#include <QFileDialog>
#include <QDateTime>
#include <QMouseEvent>

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

#include "QMenuBar"

ImageWidget::ImageWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ImageWidget),
    image(nullptr),
    show_slice_control(false),
    output_widget(this),
    inner_image_frame(nullptr),
    q_image(nullptr),
    output_widget2(this)
{
    ui->setupUi(this);

    this->ui->slice_control->setVisible(this->show_slice_control);

    connect(this, SIGNAL(fireStatusTextChange(QString)),
            this, SLOT(handleStatusTextChange(QString)));

    qRegisterMetaType<ITKImage>("ITKImage");
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

    // register modules and add widget modules
    module_parent->hide();
    module_parent->setUpdatesEnabled(false);
    uint index = 0;
    for(auto module : modules)
    {
        auto widget = dynamic_cast<BaseModuleWidget*>(module);
        if(widget != nullptr)
            module_parent->insertTab(index++, widget, module->getTitle());

        std::cout << "registering module: " << module->getTitle().toStdString() << std::endl;
        module->registerModule(this);
    }
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
        if(widget == nullptr)
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

    tgv_widget->setIterationFinishedCallback([this](uint index, uint count, ITKImage u) {
        emit this->fireStatusTextChange(QString("iteration %1 / %2").arg(
                                            QString::number(index+1),
                                            QString::number(count)));
        emit this->output_widget->fireImageChange(u.getPointer());
    });
}

ImageWidget::~ImageWidget()
{
    delete ui;
}


void ImageWidget::setImage(ITKImage image)
{
    this->q_image = nullptr; // redo converting
    this->image = image.clone();

    this->setInputRanges();
    this->setSliceIndex(0);
    this->setMinimumSizeToImage();

    emit this->imageChanged(this->image);
}

uint ImageWidget::userSliceIndex() const
{
    return this->ui->slice_slider->value();
}

void ImageWidget::setSliceIndex(uint slice_index)
{
    if(this->image.isNull())
        return;

    if(slice_index < 0 || slice_index >= this->image.getDepth())
    {
        std::cerr << "invalid slice_index for this image" << std::endl << std::flush;
        return;
    }

    image.setVisibleSliceIndex(slice_index);
    if(this->ui->slice_slider->value() != this->image.getVisibleSliceIndex())
    {
        this->ui->slice_slider->setValue(this->image.getVisibleSliceIndex());
    }
    if(this->ui->slice_spinbox->value() != this->image.getVisibleSliceIndex())
    {
        this->ui->slice_spinbox->setValue(this->image.getVisibleSliceIndex());
    }
    this->paintImage();

    if(image.getImageDimension() > 2)
        this->showSliceControl();

    emit this->sliceIndexChanged(slice_index);
}

void ImageWidget::paintImage(bool repaint)
{
    if(this->image.isNull())
        return;

    if(repaint && q_image != nullptr) {
        delete q_image;
        q_image = nullptr;
    }

    if(q_image == nullptr)
    {
        q_image = ITKToQImageConverter::convert(this->image);
        if(inner_image_frame != nullptr)
            delete inner_image_frame;
        inner_image_frame = new QLabel(this->ui->image_frame);
        inner_image_frame->setMouseTracking(true);
        inner_image_frame->installEventFilter(this);
        inner_image_frame->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        inner_image_frame->setMinimumSize(q_image->size());
        //  frame->setMaximumSize(q_image.size());
        //  frame->setBaseSize(q_image.size());
        //  frame->setFixedSize(q_image.size());

        QPixmap pixmap = QPixmap::fromImage(*q_image);
        emit this->pixmapPainted(&pixmap);  // other modules paint into it here
        inner_image_frame->setPixmap(pixmap);

        inner_image_frame->setAlignment(Qt::AlignTop | Qt::AlignLeft);

        QVBoxLayout* layout = new QVBoxLayout();
        layout->addWidget(inner_image_frame);
        if(this->ui->image_frame->layout() != nullptr)
            delete this->ui->image_frame->layout();
        this->ui->image_frame->setLayout(layout);
        this->ui->image_frame->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        this->ui->image_frame->setMinimumSize(q_image->size());

    }
}

void ImageWidget::on_slice_slider_valueChanged(int user_slice_index)
{
    this->setSliceIndex(user_slice_index);
}


void ImageWidget::showSliceControl()
{
    this->show_slice_control = true;
    this->ui->slice_control->setVisible(true);
}

void ImageWidget::connectSliceControlTo(ImageWidget* other_image_widget)
{
    connect(other_image_widget, &ImageWidget::sliceIndexChanged,
            this, &ImageWidget::connectedSliceControlChanged);
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

void ImageWidget::connectedSliceControlChanged(uint slice_index)
{
    this->setSliceIndex(slice_index);
}

void ImageWidget::mouseReleaseEvent(QMouseEvent *)
{
    std::cout << "mouse released" << std::endl;

    emit this->mouseReleasedOnImage();
}

void ImageWidget::mousePressEvent(QMouseEvent * mouse_event)
{
    if(this->image.isNull() || this->inner_image_frame == nullptr)
        return;

    QPoint position = this->inner_image_frame->mapFromGlobal(mouse_event->globalPos());
    // std::cout << "mouse pressed at " << position.x() << "|" << position.y() << std::endl;

    emit this->mousePressedOnImage(mouse_event->button(), position);
}

bool ImageWidget::eventFilter(QObject *target, QEvent *event)
{
    if(this->image.isNull())
        return false;

    if(event->type() == QEvent::Paint)
        return false;

    if(event->type() == QEvent::MouseMove && target == inner_image_frame)
    {
        QMouseEvent* mouse_event = static_cast<QMouseEvent*>(event);
        if(mouse_event == nullptr)
            return false;

        QPoint position = this->inner_image_frame->mapFromGlobal(mouse_event->globalPos());

        //std::cout << "mouse move at " << position.x() << "|" << position.y() << std::endl;

        emit this->mouseMoveOnImage(mouse_event->buttons(), position);

    }
    return false; // always returning false, so the pixmap is painted
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

void ImageWidget::setInputRanges()
{
    if(this->image.isNull())
        return;

    this->ui->slice_slider->setMinimum(0); // first slice gets slice index 0
    this->ui->slice_spinbox->setMinimum(this->ui->slice_slider->minimum());

    if(this->image.getImageDimension() >= 3)
    {
        this->ui->slice_slider->setMaximum(this->image.getDepth() - 1);
        this->ui->slice_spinbox->setMaximum(this->ui->slice_slider->maximum());
    }

}

void ImageWidget::on_slice_spinbox_valueChanged(int slice_index)
{
    if(slice_index != this->image.getVisibleSliceIndex())
    {
        this->setSliceIndex(slice_index);
    }
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

void ImageWidget::paintEvent(QPaintEvent *paint_event)
{
    QWidget::paintEvent(paint_event);
    this->paintImage();
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

void ImageWidget::handleRepaintImage()
{
    this->paintImage(true);
}
