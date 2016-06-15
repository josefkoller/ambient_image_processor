#ifndef PARAMETERVALUESET_H
#define PARAMETERVALUESET_H

#include <QWidget>

namespace Ui {
class ParameterValueSet;
}

class ParameterValueSet : public QWidget
{
    Q_OBJECT
private:
    typedef double ParameterValue;
    typedef std::pair<ParameterValue, ParameterValue> ParameterSet;
    typedef std::vector<ParameterSet> ParameterList;
public:
    explicit ParameterValueSet(QWidget *parent = 0);
    ~ParameterValueSet();

    ParameterList getParameterList() { return this->parameter_list; };
private slots:
    void on_add_button_clicked();

    void on_remove_selected_button_clicked();

    void on_add_last_divided_by_ten_button_clicked();

    void on_clear_button_clicked();

    void on_exrapolate_button_clicked();

private:
    Ui::ParameterValueSet *ui;

    ParameterList parameter_list;

    void updateOutputList();
    QString text(ParameterSet set);
};

#endif // PARAMETERVALUESET_H
