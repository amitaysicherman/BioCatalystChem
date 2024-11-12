#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --gres=gpu:A40:1
#SBATCH -c 10
#SBATCH --array=1-2

# Define the configurations as a long string with a delimiter (| in this case)
configs_1="--ec_type 0 --mix 1 |\
  --ec_type 1 --mix 1 |\
  --ec_type 2 --mix 1 |\
  --ec_type 2 --mix 1 --addec 1 |\
  --ec_type 3 --mix 1 --addec 1"

configs_2="--ec_type 2 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 --mix 1 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 --mix 1 |\
  --ec_type 2 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 --addec 1 --mix 1 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 --addec 1 --mix 1 |\
  --ec_type 3 --mix 1"

# Set the GPU memory fraction
tasks_on_gpu=5

# Choose configs based on array task ID
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    configs=$configs_1
else
    configs=$configs_2
fi

# Split the config string into an array using | as a delimiter
IFS='|' read -ra config_array <<< "$configs"

# Process the configurations
for config in "${config_array[@]}"; do
    {
        python finetune_ecreact.py $config --tasks_on_gpu $tasks_on_gpu
    } &
done

# Wait for all background jobs to finish
wait