#!/bin/sh

destination=~/Documents/sources/c_itk_qt_cuda_imagejoe
rm -rf $destination
mkdir -p $destination
cp build/output/ambient_application $destination
git log --pretty=format:'%H' -n 1 > $destination/version.txt
