#!/bin/bash
##SBATCH -p gpu_v100_2
#SBATCH -p gpu_a100_8
#SBATCH -N 1
#SBATCH -t 03:10:30
#SBATCH --job-name="Exp3"
#SBATCH -o logs/slurm.%j.out
#SBATCH -e logs/slurm.%j.err
#SBATCH --gres=gpu:1

# sbatch job.sh train-test data/CoordinationDataSet/input/train_copy.coord model_dir_03 data/CoordinationDataSet/gold/test_copy.coord data/CoordinationDataSet/output/Predictions_BART_large.txt BART
# sbatch job.sh train data/CoordinationDataSet/input/train.coord model_dir data/CoordinationDataSet/gold/test.txt data/CoordinationDataSet/output/Predictions_T5_base.txt T5
# sbatch job.sh test data/CoordinationDataSet/input/train.coord model_dir data/CoordinationDataSet/gold/test.coord data/CoordinationDataSet/output/Predictions_T5_base.txt T5
# python3 Utils/wire57.py T5 data/CoordinationDataSet/gold/test_copy.coord  data/CoordinationDataSet/output2/predictions/Prediction_T5_small_b24.coord > data/CoordinationDataSet/output2/evaluations/wire57_f1/Result_T5_small_b24_wire57v2.txt
# python3 Utils/computeRogue.py T5 data/CoordinationDataSet/output2/predictions/Prediction_T5_base_b16.coord data/CoordinationDataSet/output2/evaluations/rouge/Result_T5_base_b16_rouge.txt
# python3 Utils/preprocess.py T5 data/CoordinationDataSet/output/predictions/Prediction_T5_large.coord data/CoordinationDataSet/output/predictions/Prediction_T5_large.conj


spack load anaconda3@2022.05
conda init bash
eval "$(conda shell.bash hook)"
conda activate /home/__NAME__/.conda/envs/legalIE
cd /scratch/__NAME__/LegalIE/__NAME__/exp1
python /scratch/__NAME__/LegalIE/__NAME__/exp1/run.py $1 $2 $3 $4 $5 $6 $7 $8
