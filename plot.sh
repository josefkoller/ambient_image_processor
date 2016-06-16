#!/bin/bash

x="alpha0"
y="alpha1"
z="deshaded_entropy"
gnuplot_file="data/output/$1/gnuplot.dat"
build/output/sqlite_to_gnuplot data/output/$1/database.sqlite $gnuplot_file $x $y $z
gnuplot -e "filename='${gnuplot_file}'; x='$x'; y='$y'; z='$z'" gnuplot/plot_data.gp
