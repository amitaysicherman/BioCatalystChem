


#!/bin/bash

# Define the configurations as a long string with a delimiter (| in this case)
configs="--use_ec 1 --ec_split 1 --load_cp results_old/v4/regular/checkpoint-280000 --ecreact_only 1 --freeze_encoder 1 --post_encoder 1|\
  --use_ec 1 --ec_split 0 --lookup_len 5 --load_cp results_old/v4/regular/checkpoint-280000 --ecreact_only 1 --freeze_encoder 1 --post_encoder 1|\
  --dae 1 --ec_split 1 --lookup_len 5 --load_cp results_old/v4/regular/checkpoint-280000 --ecreact_only 1 --freeze_encoder 1 --post_encoder 1|\
  --use_ec 0 --ec_split 1 --load_cp results_old/v4/regular/checkpoint-280000 --ecreact_only 1 --freeze_encoder 1 --post_encoder 1|\
    --use_ec 1 --ec_split 1 --load_cp results_old/v4/regular/checkpoint-280000  --freeze_encoder 1 --post_encoder 1|\
  --use_ec 1 --ec_split 0 --lookup_len 5 --load_cp results_old/v4/regular/checkpoint-280000 --freeze_encoder 1 --post_encoder 1|\
  --dae 1 --ec_split 1 --lookup_len 5 --load_cp results_old/v4/regular/checkpoint-280000 --freeze_encoder 1 --post_encoder 1|\
  --use_ec 0 --ec_split 1 --load_cp results_old/v4/regular/checkpoint-280000 --freeze_encoder 1 --post_encoder 1|\
  --use_ec 1 --ec_split 1 --load_cp results_old/v4/regular/checkpoint-280000|\
  --use_ec 1 --ec_split 0 --lookup_len 5 --load_cp results_old/v4/regular/checkpoint-280000|\
  --dae 1 --ec_split 1 --lookup_len 5 --load_cp results_old/v4/regular/checkpoint-280000|\
  --use_ec 0 --ec_split 1 --load_cp results_old/v4/regular/checkpoint-280000|\
    --use_ec 1 --ec_split 1 --load_cp results_old/v4/regular/checkpoint-280000 --ecreact_only 1|\
  --use_ec 1 --ec_split 0 --lookup_len 5 --load_cp results_old/v4/regular/checkpoint-280000 --ecreact_only 1|\
  --dae 1 --ec_split 1 --lookup_len 5 --load_cp results_old/v4/regular/checkpoint-280000 --ecreact_only 1|\
  --use_ec 0 --ec_split 1 --load_cp results_old/v4/regular/checkpoint-280000 --ecreact_only 1"

# Count the number of configurations by counting the number of delimiters (|) + 1
num_configs=$(echo "$configs" | tr -cd '|' | wc -c)
num_configs=$((num_configs + 1))

# Create a temporary SLURM submission script
cat <<EOF > slurm_submit.sh
#!/bin/sh
#SBATCH --time=7-00
#SBATCH --array=1-$num_configs
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:A40:1

# Adjust SLURM_ARRAY_TASK_ID to match zero-indexed array (subtract 1 from SLURM_ARRAY_TASK_ID)
index=\$((SLURM_ARRAY_TASK_ID - 1))

# Split the long config string into an array using | as a delimiter
IFS='|' read -r -a configs <<< "$configs"

# Check if the index is valid and run the python script with the selected configuration
if [ "\$index" -ge 0 ] && [ "\$index" -lt "\${#configs[@]}" ]; then
  python train.py \${configs[\$index]}
else
  echo "Invalid SLURM_ARRAY_TASK_ID: \$SLURM_ARRAY_TASK_ID"
fi
EOF

# Submit the dynamically generated SLURM script
sbatch slurm_submit.sh
