#!/bin/bash

mkdir -p out/final/

for FILE in configs/final/ufm/C_2/*
do
echo -e "$FILE"
python gufm.py $FILE
done

for FILE in configs/final/ufm/C_4/*
do
echo -e "$FILE"
python gufm.py $FILE
done
