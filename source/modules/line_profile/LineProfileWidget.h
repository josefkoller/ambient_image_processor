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
    explicit LineProfileWidget(QWidget *parent = 0);
    ~LineProfileWidget();

    QList<LineProfile> getProfileLines() { return this->profile_lines; }
    int selectedProfileLineIndex();

    void mousePressedOnImage(Qt::MouseButton button, QPoint position);
    void connectTo(LineProfileWidget* other);
    void paintSelectedProfileLine();
private:
    Ui::LineProfileWidget *ui;

    QList<LineProfile> profile_lines;
    bool adding_profile_line;

    LineProfileWidget* profile_line_parent;

private slots:
    void line_profile_mouse_move(QMouseEvent*);
    void on_add_profile_line_button_clicked();
    void on_line_profile_list_widget_currentRowChanged(int currentRow);

    void on_line_profile_list_widget_itemSelectionChanged();

signals:
    void profileLinesChanged();
public slots:
    void connectedProfileLinesChanged();

public:
    virtual void registerModule(ImageWidget* image_widget);
};

#endif // LINEPROFILEWIDGET_H
