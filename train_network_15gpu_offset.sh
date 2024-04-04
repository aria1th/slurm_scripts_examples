#!/bin/bash
#SBATCH --job-name=continue
#SBATCH --output=multinode-o-%x.%j
#SBATCH --error=multinode-e-%x.%j
#SBATCH --partition=SMALL-GPU # partition name
#SBATCH --nodes=6                   # number of nodes
#SBATCH --gres=gpu:4              # number of GPUs per node
#SBATCH --time=72:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --cpus-per-gpu=9
#SBATCH --qos=QOS_NAME           # qos name


# example, get 2 node with 2 gpu each

######################
### Set enviroment ###
######################
# Activate your Python environment
conda init
conda activate kohya
unset LD_LIBRARY_PATH
# Change to the directory containing your script
cd /dir/sd-scripts
gpu_count=$(scontrol show job $SLURM_JOB_ID | grep -oP 'TRES=.*?gpu=\K(\d+)' | head -1)
######################
# set SLURM_JOB_NODELIST
#export SLURM_JOB_NODELIST=node[08-21]
######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO
#######################
export NCCL_ASYNC_ERROR_HANDLING=0
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1
#######################
echo "SLURM_JOB_NODELIST is $SLURM_JOB_NODELIST"
node_name=$(echo $SLURM_JOB_NODELIST | sed 's/node-list: //' | cut -d, -f1)
MASTER_ADDR=$(getent ahosts $node_name | head -n 1 | awk '{print $1}')
PORT=29507 # Port number that won't conflict with other users

export LAUNCHER="/home/usr/miniconda3/envs/kohya/bin/accelerate launch \
    --num_processes $gpu_count \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port $PORT \
    "
export SCRIPT="/dir/sd-scripts/sdxl_train_network.py "
export SCRIPT_ARGS=" \
    --config_file /dir/train_config_loKr_continue15x.toml"
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
echo "$CMD"
# call your script for all nodes
srun $CMD
