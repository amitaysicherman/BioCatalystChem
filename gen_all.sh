#!/bin/bash
#SBATCH --time=1-00
#SBATCH --array=1-30
#SBATCH --mem=128G
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --requeue

# Get the list of directories in results/
RESULTS_DIR="results_work_version/"
RUN_NAMES=($(ls -d $RESULTS_DIR*/ | xargs -n 1 basename))

# Check if the SLURM_ARRAY_TASK_ID is within the bounds of the available run names
if [ $SLURM_ARRAY_TASK_ID -le ${#RUN_NAMES[@]} ]; then\
    RUN_NAME=${RUN_NAMES[$SLURM_ARRAY_TASK_ID - 1]} # Arrays are 0-indexed
    if [[ $RUN_NAME == *"mix"* ]]; then
        echo "Skipping run: $RUN_NAME (contains 'mix')"
        exit 0
    fi
    python eval_gen.py --run_name $RUN_NAME --fast 0 --split test --dups 0 --per_level 1 --per_ds 1 --res_base ${RESULTS_DIR}  --bs 16
else
    echo "Error: SLURM_ARRAY_TASK_ID is out of bounds."
fi
