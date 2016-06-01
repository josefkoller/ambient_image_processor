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
    void mouseMoveOnImage(Qt::MouseButtons button, ITKImage::Index cursor_index);

    void connectTo(LineProfileWidget* other);
    void paintSelectedProfileLine();
private:
    Ui::LineProfileWidget *ui;

    LineProfile::Point cursor_position;
    QPointF projected_cursor_point;

    QList<LineProfile> profile_lines;

    LineProfileWidget* profile_line_parent;

    ITKImage image;

    bool setting_line_point;

    QVector<double> intensitiesQ;
    QVector<double> distancesQ;
private slots:
    void line_profile_mouse_move(QMouseEvent*);
    void on_add_profile_line_button_clicked();

    void on_line_profile_list_widget_itemSelectionChanged();

    void paintSelectedProfileLineInImage(QPixmap* pixmap);
    void on_setting_line_point_button_clicked();

    void on_paint_parent_lines_checkbox_toggled(bool checked);

signals:
    void profileLinesChanged();
public slots:
    void connectedProfileLinesChanged();

public:
    virtual void registerModule(ImageWidget* image_widget);
    virtual void connectTo(BaseModule* other);

private:
    static const QColor line_color;
    static const QColor line_with_parent_color;
    static const QColor cursor_color;
    static const QColor start_point_color;
    static const QColor end_point_color;
};

#endif // LINEPROFILEWIDGET_H
