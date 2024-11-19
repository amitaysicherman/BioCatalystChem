#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --gres=gpu:A40:1
#SBATCH -c 8
#SBATCH --array=1-14

configs="--ec_type 0  --max_length 200 |\
  --ec_type 1  --max_length 205  |\
  --ec_type 2  --max_length 200  |\
  --ec_type 3  --max_length 200  |\
  --ec_type 2 --prequantization 1 --n_hierarchical_clusters 7 --n_pca_components 0 --n_clusters_pca 7 --addec 1 --max_length 213  |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 7 --n_pca_components 0 --n_clusters_pca 7 --addec 1 --max_length 213 |\
  --ec_type 2 --prequantization 1 --n_hierarchical_clusters 7 --n_pca_components 0 --n_clusters_pca 7  --max_length 208  |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 7 --n_pca_components 0 --n_clusters_pca 7  --max_length 208  |\
  --ec_type 2 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 7 --n_clusters_pca 7 --addec 1 --max_length 213  |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 7 --n_clusters_pca 7 --addec 1 --max_length 213 |\
  --ec_type 2 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 7 --n_clusters_pca 7  --max_length 208  |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 7 --n_clusters_pca 7  --max_length 208"

# Split the config string into an array using '|' as a delimiter
IFS='|' read -ra config_array <<< "$configs"

config=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}
python finetune_ecreact.py $config
