#!/bin/sh
#SBATCH --time=7-00
#SBATCH --array=1-10
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:A40:1

# Run different script based on SLURM_ARRAY_TASK_ID
case $SLURM_ARRAY_TASK_ID in
  1)
    python train.py --use_ec 0 --ec_split 1
    ;;
  2)
    python train.py --use_ec 1 --ec_split 1
    ;;
  3)
    python train.py --use_ec 1 --ec_split 0 --lookup_len 1
    ;;
  4)
    python train.py --use_ec 1 --ec_split 0 --lookup_len 5
    ;;
  5)
    python train.py --dae 1 --ec_split 1 --lookup_len 1 --seq_add 0
    ;;
  6)
    python train.py --dae 1  --ec_split 1 --lookup_len 5 --seq_add 0
    ;;
  7)
    python train.py --dae 1  --ec_split 1 --lookup_len 1 --seq_add 1
    ;;
  8)
    python train.py --dae 1  --ec_split 1 --lookup_len 5 --seq_add 1
    ;;
  9)
    python train.py --use_ec 1 --ec_split 0 --lookup_len 1 --seq_add 1
    ;;
  10)
    python train.py --use_ec 1 --ec_split 0 --lookup_len 5 --seq_add 1
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    ;;
esac
