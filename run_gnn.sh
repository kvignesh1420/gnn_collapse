#!/bin/bash

mkdir -p out/final/

# for FILE in configs/final/graphtrans/T1/*
# do
# echo -e "$FILE"
# python main.py $FILE
# done

for FILE in configs/final/gps/T1/*
do
echo -e "$FILE"
rm -rf models out
python main.py $FILE
done

# for FILE in configs/final/graphconv/C_2/*
# do
# echo -e "$FILE"
# python main.py $FILE
# done

# for FILE in configs/final/graphconv/C_4/*
# do
# echo -e "$FILE"
# python main.py $FILE
# done


# for FILE in configs/final/graphconv_hetero/C_2/*
# do
# echo -e "$FILE"
# python main.py $FILE
# done

# for FILE in configs/final/graphconv_hetero/C_4/*
# do
# echo -e "$FILE"
# python main.py $FILE
# done
