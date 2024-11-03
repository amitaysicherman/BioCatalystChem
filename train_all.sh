#!/bin/bash

# Define the configurations as a long string with a delimiter (| in this case)
configs="--ec_type 0|\
  --ec_type 1 |\
  --ec_type 2 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 |\
  --ec_type 2 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 --addec 1 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 --addec 1 |\
  --ec_type 2 |\
  --ec_type 3 |\
  --ec_type 0 --mix 1 |\
  --ec_type 1 --mix 1 |\
  --ec_type 2 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 --mix 1 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 --mix 1 |\
  --ec_type 2 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 --addec 1 --mix 1 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 --addec 1 --mix 1 |\
  --ec_type 2 --mix 1 |\
  --ec_type 3 --mix 1 |\
  --ec_type 0 --lora 1 |\
  --ec_type 1 --lora 1 |\
  --ec_type 2 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 --lora 1 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 --lora 1 |\
  --ec_type 2 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 --addec 1 --lora 1 |\
  --ec_type 3 --prequantization 1 --n_hierarchical_clusters 0 --n_pca_components 10 --n_clusters_pca 10 --addec 1 --lora 1 |\
  --ec_type 2 --lora 1 |\
  --ec_type 3 --lora 1"

# Count the number of configurations by counting the number of delimiters (|) + 1
num_configs=$(echo "$configs" | tr -cd '|' | wc -c)
num_configs=$((num_configs + 1))

cat <<EOF > slurm_submit.sh
#!/bin/bash
#SBATCH --time=7-00
#SBATCH --array=1-$num_configs
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:A4000:1

# Adjust SLURM_ARRAY_TASK_ID to match zero-indexed array (subtract 1 from SLURM_ARRAY_TASK_ID)
index=\$((SLURM_ARRAY_TASK_ID - 1))

# Split the long config string into an array using | as a delimiter
configs="$configs"
IFS='|' read -ra config_array <<< "\$configs"

# Check if the index is valid and run the python script with the selected configuration
if [ "\$index" -ge 0 ] && [ "\$index" -lt "\${#config_array[@]}" ]; then
  python finetune_ecreact.py \${config_array[\$index]}
else
  echo "Invalid SLURM_ARRAY_TASK_ID: \$SLURM_ARRAY_TASK_ID"
fi
EOF


# Submit the dynamically generated SLURM script
sbatch slurm_submit.sh
