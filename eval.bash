#!/bin/bash

for n in 03 16 24 32
do
    python3 Utils/level_scores.py T5 data/CoordinationDataSet/gold/test_copy.coord data/CoordinationDataSet/output2/predictions/Prediction_T5_base_b${n}.coord > data/CoordinationDataSet/output2/evaluations/wire57_f1_level/Result_T5_base_b${n}_wire57v3.txt
done