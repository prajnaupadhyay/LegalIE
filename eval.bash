#!/bin/bash

for n1 in 03 16 24 32
do
    for n2 in 1 2 3
    do
        python3 Utils/wire57.py BART data/CoordinationDataSet/gold/test_copy.coord data/CoordinationDataSet/outputv2/predictions/Predictions_BART_small_b${n1}_v${n2}.txt > data/CoordinationDataSet/outputv2/evaluations/wire57_f1/Result_BART_small_b${n1}_v${n2}_wire57v3.txt
    done
done