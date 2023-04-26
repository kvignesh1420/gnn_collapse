#!/bin/bash

mkdir -p out/final/

for FILE in configs/final/graphconv/*
do
echo -e "$FILE"
python main.py $FILE
done

