#!/bin/bash
#SBATCH --time=1-00
#SBATCH --array=1-30
#SBATCH --mem=64G
#SBATCH --gres=gpu:A4000:1
#SBATCH --requeue

# Get the list of directories in results/
RESULTS_DIR="results/"
RUN_NAMES=($(ls -d $RESULTS_DIR*/ | xargs -n 1 basename))

# Check if the SLURM_ARRAY_TASK_ID is within the bounds of the available run names
if [ $SLURM_ARRAY_TASK_ID -le ${#RUN_NAMES[@]} ]; then
    RUN_NAME=${RUN_NAMES[$SLURM_ARRAY_TASK_ID - 1]} # Arrays are 0-indexed
    python eval_gen.py --run_name $RUN_NAME --fast 0 --split test
else
    echo "Error: SLURM_ARRAY_TASK_ID is out of bounds."
fi
