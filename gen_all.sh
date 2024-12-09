#!/bin/bash
#SBATCH --time=1-00
#SBATCH --array=1-12
#SBATCH --mem=128G
#SBATCH -c 4
#SBATCH --gres=gpu:L40:1
#SBATCH --requeue

# Get the list of directories in results/ containing "mix"
RESULTS_DIR="results/"

PREDEFINED_NAMES=(
 'PAPER_nmix',
 'PRETRAINED_1_nmix',
 'PRETRAINED_2_nmix',
 'NO_EC_nmix',
 'PRETRAINED_3_nmix',
 'PRETRAINED_0_nmix',
)

# Assign PREDEFINED_NAMES to RUN_NAMES
RUN_NAMES=("${PREDEFINED_NAMES[@]}")

# Calculate the 0-based index for the current task
INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# Determine the split based on the index
if [ "$INDEX" -lt 6 ]; then
  SPLIT="valid"
else
  SPLIT="test"
  INDEX=$((INDEX - 6)) # Adjust the index if greater than 6
fi

RUN_NAME=${RUN_NAMES[$INDEX]}

# Run the Python evaluation script
python eval_gen_v2.py --run_name $RUN_NAME --res_base ${RESULTS_DIR} --bs 16 --split $SPLIT --remove_stereo 1 --last_only 1
