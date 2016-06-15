#include "ParameterValueSet.h"
#include "ui_ParameterValueSet.h"

ParameterValueSet::ParameterValueSet(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ParameterValueSet)
{
    ui->setupUi(this);
}

ParameterValueSet::~ParameterValueSet()
{
    delete ui;
}

void ParameterValueSet::on_add_button_clicked()
{
    ParameterSet set;
    set.first = this->ui->alpha0_spinbox->value();
    set.second = this->ui->alpha1_spinbox->value();

    this->parameter_list.push_back(set);
    this->updateOutputList();
}

void ParameterValueSet::updateOutputList()
{
    this->ui->values_listwidget->clear();
    for(auto parameter_set : this->parameter_list)
    {
        QString text = this->text(parameter_set);
        this->ui->values_listwidget->addItem(text);
    }
}

QString ParameterValueSet::text(ParameterSet set)
{
    return QString("alpha0: %1; alpha1: %2").arg(
      QString::number(set.first), QString::number(set.second) );
}

void ParameterValueSet::on_remove_selected_button_clicked()
{
    int selected_index = this->ui->values_listwidget->currentRow();
    if(selected_index >= 0 && selected_index < this->parameter_list.size() &&
            this->parameter_list.size() > 0)
    {
        this->parameter_list.erase(this->parameter_list.begin() + selected_index);
        this->updateOutputList();
    }
}

void ParameterValueSet::on_add_last_divided_by_ten_button_clicked()
{
    int last_index = this->parameter_list.size() - 1;
    if(last_index < 0)
        return;
    auto last = this->parameter_list[last_index];

    auto set = ParameterSet();
    set.first = last.first / 10.0;
    set.second = last.second / 10.0;

    this->parameter_list.push_back(set);
    this->updateOutputList();
}

void ParameterValueSet::on_clear_button_clicked()
{
    this->parameter_list.clear();
    this->updateOutputList();
}

void ParameterValueSet::on_exrapolate_button_clicked()
{
    if(this->parameter_list.size() < 2)
        return;

    auto last = this->parameter_list[this->parameter_list.size() - 1];
    auto previous = this->parameter_list[this->parameter_list.size() - 2];

    auto set = ParameterSet();
    set.first = 2*last.first - previous.first;
    set.second = 2*last.second - previous.second;

    this->parameter_list.push_back(set);
    this->updateOutputList();

}
