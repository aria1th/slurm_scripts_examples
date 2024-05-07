#!/bin/bash
#SBATCH --job-name=big_train
#SBATCH --output=multinode-o-%x.%j
#SBATCH --error=multinode-e-%x.%j
#SBATCH --partition=big_suma_rtx3090
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --cpus-per-gpu=16
#SBATCH --qos=big_qos
#SBATCH --no-kill # do not kill the job when it fails
# example, get 2 node with 2 gpu each
echo "Checking NVIDIA GPU status on each node..."
echo "-----------------------------------------"

# Loop through each node in the job's node list
for node in $(scontrol show hostnames $SLURM_JOB_NODELIST)
do
    echo "Node: $node"
    echo "-----------------------------------------"
    # Run nvidia-smi on the node and capture the output
    srun --nodes=1 --ntasks=1 --exclusive -w $node nvidia-smi
    echo "-----------------------------------------"
done
######################
### Set enviroment ###
######################
# Activate your Python environment
conda init
conda activate <env_name>
unset LD_LIBRARY_PATH
# Change to the directory containing your script
cd <path_to_script>
gpu_count=$(scontrol show job $SLURM_JOB_ID | grep -oP 'TRES=.*?gpu=\K(\d+)' | head -1)
######################
# set SLURM_JOB_NODELIST
#export SLURM_JOB_NODELIST=node[08-21]
######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO
#######################
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
export CUDA_LAUNCH_BLOCKING=1
#######################
expand_nodes() {
    # The input is something like "ip-10-0-231-[1,86]"
    local nodelist=$1

    # Replace '[' and ']' with space and split the string
    local base=$(echo $nodelist | sed -E 's/[([0-9]+),([0-9]+)]/ \1 \2 /')

    # Read into array
    read -a parts <<< "$base"

    # Check if we have three parts: prefix, start, end
    if [ ${#parts[@]} -eq 3 ]; then
        local prefix=${parts[0]}
        local start=${parts[1]}
        local end=${parts[2]}

        # Generate sequence
        for i in $(seq $start $end); do
            echo "${prefix}${i}"
            return # Return after first IP to mimic head node behavior
        done
    else
        # If the format does not include a range, just echo the input
        echo $nodelist
    fi
}
echo "SLURM_JOB_NODELIST is $SLURM_JOB_NODELIST"
node_name=$(echo $SLURM_JOB_NODELIST | sed 's/node-list: //' | cut -d, -f1)
MASTER_ADDR=$(getent ahosts $node_name | head -n 1 | awk '{print $1}')

export PORT=29509
export SCRIPT="/scratch/sd-scripts/sdxl_train_network.py "
export SCRIPT_ARGS=" \
    --config_file /scratch/train_config.toml"
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
# for each nodes, set machine_rank int and launch
# node rank = 0, 1, 2, 3, 4, 5, 6, 7...
NODE_RANK=0

is_single_node_as_bool=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l) # 1 or 2, 1 means single node

for node in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
    export RANK=0
    export LOCAL_RANK=0
    export WORLD_SIZE=$gpu_count
    export MASTER_ADDR=$head_node_ip
    export MASTER_PORT=$PORT
    export NODE_RANK=$NODE_RANK
    # if multi node, set RANK, LOCAL_RANK and backends
    if [ $is_single_node_as_bool -eq 1 ]; then
        export LAUNCHER="/scratch/miniconda3/envs/kohya/bin/accelerate launch \
            --num_processes $gpu_count"
    else
        export RANK=$NODE_RANK
        export LOCAL_RANK=$NODE_RANK
        export LAUNCHER="/scratch/miniconda3/envs/kohya/bin/accelerate launch \
            --num_processes $gpu_count \
            --num_machines $SLURM_NNODES \
            --rdzv_backend c10d \
            --main_process_ip $head_node_ip \
            --main_process_port $PORT \
            --machine_rank $NODE_RANK"
    fi
    echo "node: $node, rank: $RANK, local_rank: $LOCAL_RANK, world_size: $WORLD_SIZE, master_addr: $MASTER_ADDR, master_port: $MASTER_PORT, node_rank: $NODE_RANK"
    NODE_RANK=$((NODE_RANK+1))
    CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
    echo "CMD: $CMD"
    # kill process 
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 $CMD &
done

wait