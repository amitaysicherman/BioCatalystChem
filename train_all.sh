#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=512G
#SBATCH --requeue
#SBATCH --gres=gpu:A100:1

# Define the configurations as a long string with a delimiter (| in this case)
configs="--ec_type 0 --mix 1 |\
  --ec_type 1 --mix 1 |\
  --ec_type 2 --mix 1 |\
  --ec_type 2 --mix 1 --addec 1 |\
  --ec_type 3 --mix 1 --addec 1 |\
  --ec_type 2 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 --mix 1 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 --mix 1 |\
  --ec_type 2 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 --addec 1 --mix 1 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 --addec 1 --mix 1 |\
  --ec_type 3 --mix 1"

# Set the GPU memory fraction (adjust as needed)
memory_fraction=0.1

# Split the long config string into an array using | as a delimiter
IFS='|' read -ra config_array <<< "$configs"

# Loop through each configuration and run it in the background
for config in "${config_array[@]}"; do
  {
    # Set the GPU memory limit within each process and run the Python script
    python finetune_ecreact.py $config --tasks_on_gpu $memory_fraction
  } &
done

# Wait for all background jobs to finish before exiting
wait
