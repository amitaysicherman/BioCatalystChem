#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --gres=gpu:L40:1
#SBATCH -c 8
#SBATCH --array=1-8

configs="--ec_type 2 --daa_type 3 --mix 0|\
          --ec_type 2 --daa_type 3 --mix 0 --emb_dropout 0.1|\
          --ec_type 2 --daa_type 3 --mix 0 --emb_dropout 0.3|\
          --ec_type 2 --daa_type 3 --mix 0 --emb_dropout 0.5|\
          --ec_type 2 --daa_type 3 --mix 0 --n_bottlenecks 1|\
          --ec_type 2 --daa_type 3 --mix 0 --n_bottlenecks 4|\
          --ec_type 2 --daa_type 3 --mix 0 --n_bottlenecks 4 --emb_dropout 0.3"

# Split the config string into an array using '|' as a delimiter
IFS='|' read -ra config_array <<< "$configs"

config=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}
python finetune_ecreact_v2.py $config
