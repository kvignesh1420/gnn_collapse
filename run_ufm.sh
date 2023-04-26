#!/bin/bash

mkdir -p out/final/

for FILE in configs/final/ufm/*
do
echo -e "$FILE"
python dummy_ufm.py $FILE
done

