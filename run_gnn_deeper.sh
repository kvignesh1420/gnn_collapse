#!/bin/bash

mkdir -p out/final/

for FILE in configs/final/graphconv/L_64/*
do
echo -e "$FILE"
python main.py $FILE
done

for FILE in configs/final/graphconv/L_128/*
do
echo -e "$FILE"
python main.py $FILE
done

for FILE in configs/final/graphconv_hetero/L_64/*
do
echo -e "$FILE"
python main.py $FILE
done

for FILE in configs/final/graphconv_hetero/L_128/*
do
echo -e "$FILE"
python main.py $FILE
done