#!/bin/sh
#SBATCH --time=1-00
#SBATCH --array=1-10
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:1

# dae_1_add  dae_1_seq  dae_5_add  dae_5_seq  paper  pretrained_1  pretrained_5  regular
 case $SLURM_ARRAY_TASK_ID in
  1)
    python eval_gen.py --run_name dae_1_add
    ;;
  2)
    python eval_gen.py --run_name dae_1_seq
    ;;
  3)
    python eval_gen.py --run_name dae_5_add
    ;;
  4)
    python eval_gen.py --run_name dae_5_seq
    ;;
  5)
    python eval_gen.py --run_name paper
    ;;
  6)
    python eval_gen.py --run_name pretrained_1_add
    ;;
  7)
    python eval_gen.py --run_name pretrained_5_add
    ;;
  8)
    python eval_gen.py --run_name pretrained_1_seq
    ;;
  9)
    python eval_gen.py --run_name pretrained_5_seq
    ;;
  10)
    python eval_gen.py --run_name regular
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    ;;
  esac
# End of file