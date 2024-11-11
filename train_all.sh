#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=512G
#SBATCH --requeue
#SBATCH --gres=gpu:A100:1
#SBATCH -c 40

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

# Configuration
NUM_PROCESSES=10
TOTAL_MEMORY=512  # GB
MEMORY_PER_PROCESS=$((TOTAL_MEMORY / NUM_PROCESSES))
GPU_MEMORY_FRACTION=10

# Get number of CPU cores
NUM_CORES=$(nproc)
CORES_PER_PROCESS=$((NUM_CORES / NUM_PROCESSES))

# Split the long config string into an array using | as a delimiter
IFS='|' read -ra config_array <<< "$configs"

# Create a directory for process monitoring
mkdir -p process_logs

# Function to run a single process with resource limits
run_process() {
    local config="$1"
    local process_num="$2"

    # Calculate CPU core range for this process
    local start_core=$((process_num * CORES_PER_PROCESS))
    local end_core=$((start_core + CORES_PER_PROCESS - 1))

    # Set memory limit (in KB)
    local mem_limit=$((MEMORY_PER_PROCESS * 1024 * 1024))

    # Create a subshell to set the memory limit
    (
        # Set memory limit
        ulimit -v ${mem_limit}

        # Run the process with CPU affinity
        taskset -c ${start_core}-${end_core} python finetune_ecreact.py $config \
            --tasks_on_gpu $GPU_MEMORY_FRACTION \
            2>&1 | tee "process_logs/process_${process_num}.log"
    ) &

    # Store PID for monitoring
    echo $! > "process_logs/pid_${process_num}"
}

# Launch processes with resource limits
for i in "${!config_array[@]}"; do
    run_process "${config_array[$i]}" $i

    # Add small delay between launches to prevent resource contention
    sleep 2
done

wait # Wait for all processes to finish

