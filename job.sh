#!/bin/bash
##SBATCH -p gpu_v100_2
#SBATCH -p gpu_a100_8
#SBATCH -N 1
#SBATCH -t 03:10:30
#SBATCH --job-name="Exp3"
#SBATCH -o logs/slurm.%j.out
#SBATCH -e logs/slurm.%j.err
#SBATCH --gres=gpu:1

# usage sbatch job.sh train data/CoordinationDataSet/train.txt model_dir data/CoordinationDataSet/test.txt data/Predictions_T5_Coordination.txt T5

spack load anaconda3@2022.05
conda init bash
eval "$(conda shell.bash hook)"
conda activate /home/prajna/.conda/envs/legalIE
cd /scratch/prajna/LegalIE/sankalp/exp1
python /scratch/prajna/LegalIE/sankalp/exp1/run.py $1 $2 $3 $4 $5 $6
