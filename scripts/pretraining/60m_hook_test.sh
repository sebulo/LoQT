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

# --nproc_per_node 1 is number of GPUs per node
torchrun --standalone --nproc_per_node 2 --nnodes 1 torchrun_main_hook.py \
    --model_config configs/llama_60m.json \
    --seed 42 \
    --lr 0.01 \
    --rank 128 \
    --lora_alpha 0.4 \
    --update_proj_gap 100 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --eval_every 1000 \
    --save_every 10000 \
    --dtype bfloat16 \
    --optimizer adamw \
    --use_loqt True\
    --proj_gap_progression "exponential" \
    --increment_size 1.2 \
    --save_original_model False \
    --only_train_lora False \
    --name 60m_LoQT
    # --quantize_w '4bit' \
    # --quantize_projection_matrix '4bit' \
    # --compensate_quant_error_iterations 5 \
