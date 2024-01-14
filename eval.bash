#!/bin/bash

for n1 in 03 16 24 32
do
    for n2 in 2 3
    do
        # python3 Utils/wire57.py BART data/CoordinationDataSet/gold/test_copy.coord data/CoordinationDataSet/outputv2/predictions/Predictions_BART_small_b${n1}_v${n2}.txt > data/CoordinationDataSet/outputv2/evaluations/wire57_f1/Result_BART_small_b${n1}_v${n2}_wire57v3.txt
        # python3 Utils/overlap_score.py T5 data/SubordinationDataSet/gold/test_reduced_IP.txt data/SubordinationDataSet/output/predictions/Predictions_T5_small_bs${n1}.coords > data/SubordinationDataSet/output/evaluations/wire57_f1/Result_T5_small_b${n1}_wire57v2.txt
        # python3 Utils/overlap_score.py BART data/SubordinationDataSet/gold/test_reduced_IP.txt data/SubordinationDataSet/outputv2/predictions/Predictions_BART_small_b32_v${n2}.txt > data/SubordinationDataSet/outputv2/evaluations/wire57_f1/Result_BART_small_b32_v${n2}_wire57v2.txt
        # python3 Utils/computeRogue.py T5 data/CoordinationDataSet/outputv2/predictions/Predictions_T5_small_b${n1}_v${n2}.txt data/CoordinationDataSet/outputv2/evaluations/rouge/Result_T5_small_b${n1}_v${n2}_rouge.txt
        # python3 Utils/overlap_score.py T5 data/SubordinationDataSet/gold/test_reduced_IP.txt data/SubordinationDataSet/output/predictions/Predictions_graphene.coords > data/SubordinationDataSet/output/evaluations/wire57_f1/Result_graphene_wire57.txt
        # python3 Utils/wire57.py BART data/CoordinationDataSet/gold/test_copy.coord data/CoordinationDataSet/outputv2/predictions/Predictions_BART_small_b${n1}_v${n2}.txt > data/CoordinationDataSet/outputv2/evaluations/wire57_f1/Result_BART_small_b${n1}_v${n2}_wire57v3.txt
        python3 Utils/level_scores.py BART data/CoordinationDataSet/gold/test_copy.coord data/CoordinationDataSet/outputv2/predictions/Predictions_BART_base_b${n1}_v${n2}.txt > data/CoordinationDataSet/outputv2/evaluations/wire57_f1_level/Result_BART_base_b${n1}_v${n2}_wire57v3
    done
done