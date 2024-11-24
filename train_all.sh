#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --gres=gpu:A40:1
#SBATCH -c 8
#SBATCH --array=1-6

configs=" --ec_type 3  --max_length 213 --addec 1 --daev2 1  --alpha 50 --prequantization 1 |\
  --ec_type 3  --max_length 208 --daev2 1  --alpha 50 --prequantization 1  |\
  --ec_type 3  --max_length 213 --addec 1 --daev2 1 --alpha 10 --prequantization 1 |\
  --ec_type 3  --max_length 208 --daev2 1  --alpha 10 --prequantization 1  |\
  --ec_type 3  --max_length 213 --addec 1 --daev2 1  --alpha 90 --prequantization 1  |\
  --ec_type 3  --max_length 208 --daev2 1 --alpha 90 --prequantization 1 "
# Split the config string into an array using '|' as a delimiter
IFS='|' read -ra config_array <<< "$configs"

config=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}
python finetune_ecreact.py $config
