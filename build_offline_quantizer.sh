#!/bin/bash
#SBATCH --time=7-00
#SBATCH --array=1-4
#SBATCH --mem=256G
#SBATCH -c 8
#SBATCH --requeue

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p build_offline_quantizer.txt)
python offline_quantizer.py $args