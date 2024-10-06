#!/bin/sh
#SBATCH --time=7-00
#SBATCH --array=1-4
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:A40:1

# Run different script based on SLURM_ARRAY_TASK_ID
case $SLURM_ARRAY_TASK_ID in
  1)
    python train.py --dae 1 --load_cp 'results/regular/checkpoint-495000' --ec_split 1 --lookup_len 1 --seq_add 0 --ecreact_only 1
    ;;
  2)
    python train.py --dae 1 --load_cp 'results/regular/checkpoint-495000' --ec_split 1 --lookup_len 5 --seq_add 0 --ecreact_only 1
    ;;
  3)
    python train.py --dae 1 --load_cp 'results/regular/checkpoint-495000' --ec_split 1 --lookup_len 1 --seq_add 1 --ecreact_only 1
    ;;
  4)
    python train.py --dae 1 --load_cp 'results/regular/checkpoint-495000' --ec_split 1 --lookup_len 5 --seq_add 1 --ecreact_only 1
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    ;;
esac
