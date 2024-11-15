#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --gres=gpu:A40:1
#SBATCH -c 10
#SBATCH --array=1-1

configs="--ec_type 3  --max_length 200 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 3  --addec 1 --max_length 205 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 3  --max_length 200 --alpha 10 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 3  --addec 1 --max_length 205 --alpha 10 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 3  --max_length 200 --alpha 90 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 3  --addec 1 --max_length 205 --alpha 90 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 0  --max_length 200 --regpre 1 --dups 3 --use_bs 256  |\
  --ec_type 1  --max_length 205 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 2  --max_length 200 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 2  --addec 1 --max_length 205 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 2 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 7 --n_clusters_pca 7  --max_length 208 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 2 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 7 --n_clusters_pca 7 --addec 1 --max_length 213 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 2 --prequantization 1 --n_hierarchical_clusters 7 --n_pca_components 0 --n_clusters_pca 7  --max_length 208 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 2 --prequantization 1 --n_hierarchical_clusters 7 --n_pca_components 0 --n_clusters_pca 7 --addec 1 --max_length 213 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 7 --n_clusters_pca 7  --max_length 208 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 7 --n_clusters_pca 7 --addec 1 --max_length 213 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 7 --n_pca_components 0 --n_clusters_pca 7  --max_length 208 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 7 --n_pca_components 0 --n_clusters_pca 7 --addec 1 --max_length 213 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 7 --n_clusters_pca 7  --max_length 208 --alpha 10 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 7 --n_clusters_pca 7 --addec 1 --max_length 213 --alpha 10 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 7 --n_pca_components 0 --n_clusters_pca 7  --max_length 208 --alpha 10 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 7 --n_pca_components 0 --n_clusters_pca 7 --addec 1 --max_length 213 --alpha 10 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 7 --n_clusters_pca 7  --max_length 208 --alpha 90 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 7 --n_clusters_pca 7 --addec 1 --max_length 213 --alpha 90 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 7 --n_pca_components 0 --n_clusters_pca 7  --max_length 208 --alpha 90 --regpre 1 --dups 3 --use_bs 256 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 7 --n_pca_components 0 --n_clusters_pca 7 --addec 1 --max_length 213 --alpha 90 --regpre 1 --dups 3 --use_bs 256 "

# Set the GPU memory fraction
tasks_on_gpu=6

# Split the config string into an array using '|' as a delimiter
IFS='|' read -ra config_array <<< "$configs"

# Calculate the starting index for this job based on SLURM_ARRAY_TASK_ID
start_index=$(( (SLURM_ARRAY_TASK_ID - 1) * tasks_on_gpu ))
end_index=$(( start_index + tasks_on_gpu ))

# Extract the subset of configs for this specific array job
selected_configs=("${config_array[@]:start_index:tasks_on_gpu}")

# Process each selected configuration
for config in "${selected_configs[@]}"; do
    {
        python finetune_ecreact.py $config --tasks_on_gpu $tasks_on_gpu
    } &
done

# Wait for all background jobs to finish
wait
