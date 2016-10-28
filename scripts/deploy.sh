#!/bin/sh

destination=~/Documents/sources/ambient_image_processor
rm -rf $destination
mkdir -p $destination
cp ../build/output/ambient_application $destination
git log --pretty=format:'%H' -n 1 > $destination/version.txt
