#!/bin/bash

for n in 03 16 24 32
do
    python3 Utils/level_scores.py BART data/CoordinationDataSet/gold/test_copy.coord data/CoordinationDataSet/output2/predictions/Prediction_BART_base_b${n}.coord > data/CoordinationDataSet/output2/evaluations/wire57_f1_level/Result_BART_base_b${n}_wire57v2.txt
done