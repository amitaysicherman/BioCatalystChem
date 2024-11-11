#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH --gres=gpu:A40:2
#SBATCH -c 20
#SBATCH -w newton3
#SBATCH -p newton

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
tasks_on_gpu=5  # Reduced from 10 since we're splitting across 2 GPUs

# Split the long config string into an array using | as a delimiter
IFS='|' read -ra config_array <<< "$configs"

# Calculate the midpoint to split tasks between GPUs
total_configs=${#config_array[@]}
midpoint=$((total_configs / 2))

# Function to run tasks on a specific GPU
run_on_gpu() {
    local gpu_id=$1
    local start_idx=$2
    local end_idx=$3

    for ((i=start_idx; i<end_idx; i++)); do
        config="${config_array[$i]}"
        {
            # Set specific GPU and run the Python script
            CUDA_VISIBLE_DEVICES=$gpu_id python finetune_ecreact.py $config --tasks_on_gpu $tasks_on_gpu
        } &
    done
}

# Run first half of configs on GPU 0
run_on_gpu 0 0 $midpoint

# Run second half of configs on GPU 1
run_on_gpu 1 $midpoint $total_configs

# Wait for all background jobs to finish before exiting
wait