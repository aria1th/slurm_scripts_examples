#!/bin/bash
#SBATCH --job-name=flux512
#SBATCH --output=flux512-o-%x.%j
#SBATCH --error=flux512-e-%x.%j
#SBATCH --partition=big_gpu_partition
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:6
#SBATCH --time=72:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --cpus-per-gpu=16
#SBATCH --qos=big_qos
export GPUS_PER_NODE=6

unset LD_LIBRARY_PATH
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
gpu_count=$(scontrol show job $SLURM_JOB_ID | grep -oP 'gres/gpu:[^=]+=\K\d+' | head -1)

head_node_alias=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
head_node_ip=$(ping -c 1 $head_node_alias | grep PING | awk '{print $3}' | cut -d\( -f2 | cut -d\) -f1)
export NCCL_SOCKET_IFNAME=eth0

echo "HEAD NODE IP: $head_node_ip"

export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0
echo "SLURM_JOB_NODELIST is $SLURM_JOB_NODELIST"
node_name=$(echo $SLURM_JOB_NODELIST | sed 's/node-list: //' | cut -d, -f1)
MASTER_ADDR=$(getent ahosts $node_name | head -n 1 | awk '{print $1}')
echo "MASTER ADDR: $MASTER_ADDR"
export NCCL_IB_TIMEOUT=6000
export PORT=29512

model_name="flux512_multinode"
GRADIENT_ACCUMULATION_STEPS=32
output_dir="/path/to/output/dir"
dataset_config="/path/to/dataset_config.toml"
script='/path/to/your/training_script.py'

SCRIPT_ARGS="--mixed_precision bf16 --num_cpu_threads_per_process 1 $script --pretrained_model_path /path/to/model --additional_args"

ACCELERATE="/path/to/venv/bin/accelerate"

is_single_node_as_bool=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l) # 1 or 2, 1 means single node

export LAUNCHER="$ACCELERATE launch \
    --config_file /path/to/accelerate_config.yaml \
    --num_processes $gpu_count \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --rdzv_conf timeout=10000000,rdzv_timeout=10000000 \
    --main_process_ip $head_node_ip \
    --main_process_port $PORT"

for node in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
    export RANK=0
    export LOCAL_RANK=0
    export WORLD_SIZE=$gpu_count
    export MASTER_ADDR=$head_node_ip
    export MASTER_PORT=$PORT
    export NODE_RANK=$NODE_RANK
    if [ $is_single_node_as_bool -eq 1 ]; then
        export LAUNCHER="$ACCELERATE launch \
            --config_file /path/to/accelerate_config.yaml \
            --num_processes $gpu_count"
    else
        export RANK=$NODE_RANK
        export LOCAL_RANK=$NODE_RANK
        export LAUNCHER="$ACCELERATE launch \
            --config_file /path/to/accelerate_config.yaml \
            --num_processes $gpu_count \
            --num_machines $SLURM_NNODES \
            --rdzv_backend c10d \
            --rdzv_conf timeout=10000000,rdzv_timeout=10000000 \
            --main_process_ip $head_node_ip \
            --main_process_port $PORT \
            --machine_rank $NODE_RANK"
    fi
    echo "node: $node, rank: $RANK, local_rank: $LOCAL_RANK, world_size: $WORLD_SIZE, master_addr: $MASTER_ADDR, master_port: $MASTER_PORT, node_rank: $NODE_RANK"
    NODE_RANK=$((NODE_RANK+1))
    CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
    echo "CMD: $CMD"
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --nodelist=$node $CMD &
done

wait
