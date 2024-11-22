#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --gres=gpu:L40:1
#SBATCH -c 8
#SBATCH --array=1-6

configs=" --ec_type 3  --max_length 205 --addec 1 --daev2 1  --alpha 50|\
  --ec_type 3  --max_length 200 --daev2 1  --alpha 50|\
  --ec_type 3  --max_length 205 --addec 1 --daev2 1 --alpha 10|\
  --ec_type 3  --max_length 200 --daev2 1  --alpha 10|\
  --ec_type 3  --max_length 205 --addec 1 --daev2 1  --alpha 90|\
  --ec_type 3  --max_length 200 --daev2 1 --alpha 90"
# Split the config string into an array using '|' as a delimiter
IFS='|' read -ra config_array <<< "$configs"

config=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}
python finetune_ecreact.py $config
