#!/bin/bash
##SBATCH -p gpu_v100_2
#SBATCH -p gpu_a100_8
#SBATCH -N 1
#SBATCH -t 03:10:30
#SBATCH --job-name="Exp3"
#SBATCH -o logs/slurm.%j.out
#SBATCH -e logs/slurm.%j.err
#SBATCH --gres=gpu:1

# usage sbatch job5.sh train.txt model_dir test.txt predictions.txt

spack load anaconda3
conda init bash
eval "$(conda shell.bash hook)"
conda activate /home/prajna/.conda/envs/legalIE
cd /scratch/prajna/LegalIE/exp3
python flan.py $1 $2 $3 $4 
