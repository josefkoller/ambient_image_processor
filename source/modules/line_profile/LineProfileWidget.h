#ifndef LINEPROFILEWIDGET_H
#define LINEPROFILEWIDGET_H

#include <QWidget>
#include "BaseModuleWidget.h"
#include "LineProfile.h"

namespace Ui {
class LineProfileWidget;
}

class LineProfileWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit LineProfileWidget(QString title, QWidget *parent = 0);
    ~LineProfileWidget();

    QList<LineProfile> getProfileLines() { return this->profile_lines; }
    int selectedProfileLineIndex();

    void mousePressedOnImage(Qt::MouseButton button, ITKImage::Index index);
    void connectTo(LineProfileWidget* other);
    void paintSelectedProfileLine();
private:
    Ui::LineProfileWidget *ui;

    QList<LineProfile> profile_lines;

    LineProfileWidget* profile_line_parent;

    ITKImage image;

    bool setting_line_point;
private slots:
    void line_profile_mouse_move(QMouseEvent*);
    void on_add_profile_line_button_clicked();

    void on_line_profile_list_widget_itemSelectionChanged();

    void paintSelectedProfileLineInImage(QPixmap* pixmap);
    void on_setting_line_point_button_clicked();

signals:
    void profileLinesChanged();
public slots:
    void connectedProfileLinesChanged();

public:
    virtual void registerModule(ImageWidget* image_widget);
    virtual void connectTo(BaseModule* other);
};

#endif // LINEPROFILEWIDGET_H
