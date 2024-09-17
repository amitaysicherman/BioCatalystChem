#!/bin/sh
#SBATCH --time=3-00
#SBATCH --array=1-3
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
    python train.py --use_ec 1 --ec_split 0
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    ;;
esac
