#! /bin/bash

# Script to download data

echo "Downloading prediction data from server"
file="data/prediction_data.tar.gz"

if [[ -f $file ]]; then
    echo "$file exists."
else
    echo "Downloading $file..."
    curl -s -o $file http://lfs.aminer.cn/misc/moocdata/data/prediction_data.tar.gz
    echo "Extracting files from $file..."
    tar -C data -xzvf data/prediction_data.tar.gz
    echo "Done extracting files."
fi

file="data/user_info.csv"
if [[ -f $file ]]; then
    echo "$file exists."
else
    echo "Downloading $file..."
    curl -s -o $file http://lfs.aminer.cn/misc/moocdata/data/user_info.csv
fi

file="data/course_info.csv"
if [[ -f $file ]]; then
    echo "$file exists."
else
    echo "Downloading $file..."
    curl -s -o $file http://lfs.aminer.cn/misc/moocdata/data/course_info.csv
fi

file="data/kddcup15.zip"
if [[ -f $file ]]; then
    echo "$file exists."
else
    echo "Downloading $file"
    curl -s -o $file http://lfs.aminer.cn/misc/moocdata/data/kddcup15.zip
    echo "Extracting files from $file"
    unzip -o $file -d data # unzip without asking to data folder
    echo "Done extracting files."
fi

echo "All done..."
