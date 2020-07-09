#!/bin/bash

#SBATCH -p aquila
#SBATCH --gres=gpu:1
#SBATCH -n1
#SBATCH -N1
#SBATCH -t 04:00:00
#SBATCH --mem=4GB
#SBATCH --job-name jupyter
#SBATCH --output Job%J.txt

#module purge
source ~/.bashrc
#source activate python36 

python train.py