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
#BSUB -gpu "num=2:j_exclusive=yes"   # Request 1 GPU, with exclusive access

echo "Running on $(hostname):"
nvidia-smi

eval "$(conda shell.bash hook)"
conda activate loqt

# --nproc_per_node 2 is number of GPUs per node
torchrun --standalone --nproc_per_node 2 --nnodes 1 torchrun_main.py \
    --model_config configs/llama_350m.json \
    --seed 42 \
    --lr 0.01 \
    --rank 256 \
    --lora_alpha 0.5 \
    --update_proj_gap 100 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 99 \
    --carbon_tracker_steps 99 \
    --scheduler_effective_training_steps 60000\
    --warmup_steps 6000 \
    --eval_every 2000 \
    --save_every 10000 \
    --dtype bfloat16 \
    --optimizer adamw \
    --use_loqt True\
    --quantize_w '4bit' \
    --quantize_projection_matrix '4bit' \
    --compensate_quant_error_iterations 5 \
    --proj_gap_progression "exponential" \
    --increment_size 1.2 \
    --name 350m_LoQT_carbon_tracking

torchrun --standalone --nproc_per_node 2 --nnodes 1 torchrun_main.py \
    --model_config configs/llama_350m.json \
    --seed 42 \
    --lr 0.01 \
    --rank 256 \
    --lora_alpha 0.5 \
    --update_proj_gap 100 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 99 \
    --carbon_tracker_steps 99 \
    --scheduler_effective_training_steps 60000\
    --warmup_steps 6000 \
    --eval_every 2000 \
    --save_every 10000 \
    --dtype bfloat16 \
    --optimizer galore_adamw \
    --use_loqt False\
    --name 350m_galore_carbon_tracking

torchrun --standalone --nproc_per_node 2 --nnodes 1 torchrun_main.py \
    --model_config configs/llama_350m.json \
    --seed 42 \
    --lr 0.01 \
    --rank 256 \
    --lora_alpha 0.5 \
    --update_proj_gap 100 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 99 \
    --carbon_tracker_steps 99 \
    --scheduler_effective_training_steps 60000\
    --warmup_steps 6000 \
    --eval_every 2000 \
    --save_every 10000 \
    --dtype bfloat16 \
    --optimizer adamw \
    --use_loqt False\
    --name 350m_adam_carbon_tracking

