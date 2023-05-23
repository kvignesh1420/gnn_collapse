#!/bin/bash

mkdir -p out/final/

# Bethe hessian runs L=32
for FILE in configs/final/bh/L_32/*
do
echo -e "$FILE"
python main.py $FILE
done

# Bethe hessian runs L=64
for FILE in configs/final/bh/L_64/*
do
echo -e "$FILE"
python main.py $FILE
done

# Bethe hessian runs L=128
for FILE in configs/final/bh/L_128/*
do
echo -e "$FILE"
python main.py $FILE
done

# Normalized laplacian runs L=32
for FILE in configs/final/nl/L_32/*
do
echo -e "$FILE"
python main.py $FILE
done

# Normalized laplacian runs L=64
for FILE in configs/final/nl/L_64/*
do
echo -e "$FILE"
python main.py $FILE
done


# Normalized laplacian runs L=128
for FILE in configs/final/nl/L_128/*
do
echo -e "$FILE"
python main.py $FILE
done
