#!/bin/bash
#BSUB -q p1                          # Specify queue
#BSUB -J simple_gpu_job              # Set the job name
#BSUB -n 16                          # Request number of cores (default: 1)
#BSUB -R "span[hosts=1]"             # Specify cores must be on the same host
#BSUB -R "rusage[mem=32GB]"           # Specify 4GB of memory per core/slot
#BSUB -W 72:00                       # Set walltime limit: hh:mm
#BSUB -o output_files/job.%J.out     # Specify the output file. %J is the job-id
#BSUB -e output_files/job.%J.err     # Specify the error file. %J is the job-id

# Requesting GPU resources
#BSUB -gpu "num=1:j_exclusive=yes"   # Request 1 GPU, with exclusive access

echo "Running on $(hostname):"
nvidia-smi

eval "$(conda shell.bash hook)"
conda activate loqt

# --nproc_per_node 1 is number of GPUs per node
torchrun --standalone --nproc_per_node 1 --nnodes 1 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --seed 42 \
    --lr 0.01 \
    --rank 128 \
    --lora_alpha 0.5 \
    --update_proj_gap 1000000 \
    --batch_size 512 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --eval_every 1000 \
    --dtype bfloat16 \
    --optimizer adamw \
    --use_loqt True\
    --proj_gap_progression "exponential" \
    --increment_size 1.2 \
    --name 60m_LoQT_no_update_randomP_05_orthonormal\
    --init_lora_AB_as_random_and_zeros True \
    --wandb_project 'dynamic_rank_loqt' 

# --nproc_per_node 1 is number of GPUs per node
torchrun --standalone --nproc_per_node 1 --nnodes 1 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --seed 42 \
    --lr 0.01 \
    --rank 128 \
    --lora_alpha 0.25 \
    --update_proj_gap 1000000 \
    --batch_size 512 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --eval_every 1000 \
    --dtype bfloat16 \
    --optimizer adamw \
    --use_loqt True\
    --proj_gap_progression "exponential" \
    --increment_size 1.2 \
    --name 60m_LoQT_no_update_randomP_025\
    --init_lora_AB_as_random_and_zeros True \
    --wandb_project 'dynamic_rank_loqt' 
