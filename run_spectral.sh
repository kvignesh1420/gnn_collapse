#!/bin/bash

mkdir -p out/final/

# Bethe hessian runs
for FILE in configs/final/bh/*
do
echo -e "$FILE"
python main.py $FILE
done

# Normalized laplacian runs
for FILE in configs/final/nl/*
do
echo -e "$FILE"
python main.py $FILE
done


