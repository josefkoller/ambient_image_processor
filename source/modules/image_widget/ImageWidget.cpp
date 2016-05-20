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

#include "QMenuBar"

ImageWidget::ImageWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ImageWidget),
    image(nullptr),
    slice_index(0),
    show_slice_control(false),
    output_widget(this),
    show_pixel_value_at_cursor(true),
    inner_image_frame(nullptr),
    q_image(nullptr),
    output_widget2(this)
{
    ui->setupUi(this);

    this->ui->slice_control->setVisible(this->show_slice_control);

    connect(this, SIGNAL(fireStatusTextChange(QString)),
            this, SLOT(handleStatusTextChange(QString)));

    qRegisterMetaType<Image::Pointer>("Image::Pointer");
    connect(this, SIGNAL(fireImageChange(Image::Pointer)),
            this, SLOT(handleImageChange(Image::Pointer)));


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

    // add modules
    module_parent->hide();
    module_parent->setUpdatesEnabled(false);
    uint index = 0;
    for(auto module : modules)
    {
        module_parent->insertTab(index++, module, module->getTitle());
    }
    module_parent->setUpdatesEnabled(true);
    module_parent->show();

    // create menu
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
        QAction* module_action = tools_menu->addAction(module->getTitle());
        this->connect(module_action, &QAction::triggered, this, [this, module]() {
            this->ui->operations_panel->setCurrentWidget(module);
        });
    }
    menu_bar->addMenu(tools_menu);

    this->layout()->setMenuBar(menu_bar);

    // register modules
    auto module_widgets = this->findChildren<BaseModuleWidget*>();
    for(auto module_widget : module_widgets)
    {
        std::cout << "registering module: " << module_widget->metaObject()->className() << std::endl;
        module_widget->registerModule(this);
    }

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


void ImageWidget::setImage(const Image::Pointer& image)
{
    this->q_image = nullptr; // redo converting
    this->image = ITKImageProcessor::cloneImage(image);

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
    if(this->image.IsNull())
        return;

    if(slice_index < 0 ||
            slice_index >= this->image->GetLargestPossibleRegion().GetSize()[2])
    {
        std::cerr << "invalid slice_index for this image" << std::endl << std::flush;
        return;
    }


    this->slice_index = slice_index;
    if(this->ui->slice_slider->value() != this->slice_index)
    {
        this->ui->slice_slider->setValue(this->slice_index);
    }
    if(this->ui->slice_spinbox->value() != this->slice_index)
    {
        this->ui->slice_spinbox->setValue(this->slice_index);
    }
    this->paintImage();

    emit this->sliceIndexChanged(slice_index);
}

void ImageWidget::paintImage(bool repaint)
{
    if(this->image.IsNull())
        return;

    if(repaint && q_image != nullptr) {
        delete q_image;
        q_image = nullptr;
    }

    if(q_image == nullptr)
    {
        q_image = ITKToQImageConverter::convert(this->image,
                                                this->slice_index);
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

BaseModuleWidget* ImageWidget::getModuleByName(QString module_title) const
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
    if(this->image.IsNull() || this->inner_image_frame == nullptr)
        return;

    QPoint position = this->inner_image_frame->mapFromGlobal(mouse_event->globalPos());
    // std::cout << "mouse pressed at " << position.x() << "|" << position.y() << std::endl;

    emit this->mousePressedOnImage(mouse_event->button(), position);
}

bool ImageWidget::eventFilter(QObject *target, QEvent *event)
{
    if(this->image.IsNull())
        return false;

    if(event->type() == QEvent::Paint)
        return false;

    if(event->type() == QEvent::MouseMove && target == inner_image_frame)
    {
        QMouseEvent* mouse_event = static_cast<QMouseEvent*>(event);
        if(mouse_event == nullptr || this->show_pixel_value_at_cursor == false)
            return false;

        QPoint position = this->inner_image_frame->mapFromGlobal(mouse_event->globalPos());

        //std::cout << "mouse move at " << position.x() << "|" << position.y() << std::endl;

        Image::SizeType size = this->image->GetLargestPossibleRegion().GetSize();
        if(position.x() < 0 || position.x() > size[0] ||
                position.y() < 0 || position.y() > size[1] )
        {
            return false;
        }
        Image::IndexType index;
        index[0] = position.x();
        index[1] = position.y();
        if(InputDimension > 2)
            index[2] = this->slice_index;

        // showing pixel value...
        Image::PixelType pixel_value = this->image->GetPixel(index);
        QString text = QString("pixel value at ") +
                QString::number(position.x()) +
                " | " +
                QString::number(position.y()) +
                " = " +
                QString::number(pixel_value);
        this->handleStatusTextChange(text);

        emit this->mouseMoveOnImage(mouse_event->buttons(), position);

    }
    return false; // always returning false, so the pixmap is painted
}



void ImageWidget::on_load_button_clicked()
{
    QString file_name = QFileDialog::getOpenFileName(this, "open volume file");
    if(file_name == QString::null || !QFile(file_name).exists())
    {
        return;
    }
    Image::Pointer image = ITKImageProcessor::read(file_name.toStdString());
    this->setImage(image);
}

void ImageWidget::on_save_button_clicked()
{
    if(this->image.IsNull())
        return;

    QString file_name = QFileDialog::getSaveFileName(this, "save volume file");
    if(file_name.isNull())
    {
        return;
    }
    ITKImageProcessor::write(this->image, file_name.toStdString());
}

void ImageWidget::setInputRanges()
{
    if(this->image.IsNull())
        return;

    Image::RegionType region = image->GetLargestPossibleRegion();
    Image::SizeType size = region.GetSize();

    this->ui->slice_slider->setMinimum(0); // first slice gets slice index 0
    this->ui->slice_spinbox->setMinimum(this->ui->slice_slider->minimum());

    if(size.GetSizeDimension() >= 3)
    {
        this->ui->slice_slider->setMaximum(size[2] - 1);
        this->ui->slice_spinbox->setMaximum(this->ui->slice_slider->maximum());
    }

}

void ImageWidget::on_slice_spinbox_valueChanged(int slice_index)
{
    if(slice_index != this->slice_index)
    {
        this->setSliceIndex(slice_index);
    }
}

void ImageWidget::setMinimumSizeToImage()
{
    if(this->image.IsNull())
        return;

    const int border = 50;
    this->ui->image_frame->setMinimumSize(this->image->GetLargestPossibleRegion().GetSize()[0] + border*2,
            this->image->GetLargestPossibleRegion().GetSize()[1]+border*2);
}

void ImageWidget::hidePixelValueAtCursor()
{
    this->show_pixel_value_at_cursor = false;
}

void ImageWidget::showPixelValueAtCursor()
{
    this->show_pixel_value_at_cursor = true;
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

void ImageWidget::handleImageChange(Image::Pointer image)
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
