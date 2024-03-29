#!/bin/bash

mkdir -p out/final/


for FILE in configs/final/graphconv/C_2/*
do
echo -e "$FILE"
python main.py $FILE
done

for FILE in configs/final/graphconv/C_4/*
do
echo -e "$FILE"
python main.py $FILE
done


for FILE in configs/final/graphconv_hetero/C_2/*
do
echo -e "$FILE"
python main.py $FILE
done

for FILE in configs/final/graphconv_hetero/C_4/*
do
echo -e "$FILE"
python main.py $FILE
done
