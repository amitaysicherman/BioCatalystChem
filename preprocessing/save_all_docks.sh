#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --array=1-1

#configs="--alpha 10 --v2 1|\
#  --alpha 50 --v2 1|\
#  --alpha 90 --v2 1"
#
#IFS='|' read -ra config_array <<< "$configs"
#
#config=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}
#PYTHONPATH=./ python preprocessing/save_all_docks.py $config

PYTHONPATH=./ python preprocessing/save_all_docks.py
