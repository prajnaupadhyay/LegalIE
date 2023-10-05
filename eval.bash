#!/bin/bash

for n in 03 16 24 32
do
    python3 Utils/wire57.py BART data/CoordinationDataSet/gold/test_copy.coord data/CoordinationDataSet/output2/predictions/Prediction_BART_base_b${n}_l1.coord > data/CoordinationDataSet/output2/evaluations/wire57_f1/Result_BART_base_b${n}_l1_wire57v2.txt
done