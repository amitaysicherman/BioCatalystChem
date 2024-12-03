#!/bin/bash
#SBATCH --time=1-00
#SBATCH --array=1-10
#SBATCH --mem=128G
#SBATCH -c 4
#SBATCH --gres=gpu:L40:1
#SBATCH --requeue

# Get the list of directories in results/ containing "mix"
RESULTS_DIR="results/"

PREDEFINED_NAMES=(
    "PRETRAINED_1"
    "NO_EC"
    "PAPER"
    "PRETRAINED_2"
    "PRETRAINED_0"
    "PRETRAINED_3"
    "PRETRAINED_0_ec"
    "PRETRAINED_1_ec"
    "PRETRAINED_3_ec"
    "PRETRAINED_2_ec"
)

# Assign PREDEFINED_NAMES to RUN_NAMES
RUN_NAMES=("${PREDEFINED_NAMES[@]}")

RUN_NAME=${RUN_NAMES[$SLURM_ARRAY_TASK_ID - 1]} # Arrays are 0-indexed

python eval_gen_v2.py --run_name $RUN_NAME --res_base ${RESULTS_DIR} --bs 16 --split train --sample_size 5000
python eval_gen_v2.py --run_name $RUN_NAME --res_base ${RESULTS_DIR} --bs 16 --split valid
python eval_gen_v2.py --run_name $RUN_NAME --res_base ${RESULTS_DIR} --bs 16 --split test