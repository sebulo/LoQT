#!/bin/bash

#BSUB -q p1                          # Specify queue
#BSUB -J c4loqt              # Set the job name
#BSUB -n 16                          # Request number of cores (default: 1)
#BSUB -R "span[hosts=1]"             # Specify cores must be on the same host
#BSUB -R "rusage[mem=32GB]"           # Specify 4GB of memory per core/slot
#BSUB -W 72:00                       # Set walltime limit: hh:mm
##BSUB -u your_email_address         # Set the email address (uncomment to use)
#BSUB -B                             # Send notification at start
#BSUB -N                             # Send notification at completion
#BSUB -o job.%J.out     # Specify the output file. %J is the job-id
#BSUB -e job.%J.err     # Specify the error file. %J is the job-id

# Requesting GPU resources
#BSUB -gpu "num=2:j_exclusive=yes"   # Request 1 GPU, with exclusive access



# LoQT COLA r=32
python run_glue.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name cola \
  --num_train_epochs 20 \
  --seed 9876 \
  --lora_r 32 \
  --lora_alpha 2 \
  --update_proj_gap 10 \
  --bnb_4bit_quant_type nf4 \
  --quantize_w '4bit' \
  --quantize_projection_matrix '4bit' \
  --compensate_quant_error_iterations 5 \
  --max_length 256 \
  --pad_to_max_length \
  --per_device_train_batch_size 32 \
  --learning_rate 8e-5 \
  --output_dir checkpoints \
  --use_loqt true \
  --single_gpu \
  --with_tracking \
  --report_to wandb 

# LoQT-nq COLA r=32
python run_glue.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name cola \
  --num_train_epochs 20 \
  --seed 9876 \
  --lora_r 32 \
  --lora_alpha 2 \
  --update_proj_gap 2400 \
  --max_length 256 \
  --pad_to_max_length \
  --per_device_train_batch_size 32 \
  --learning_rate 8e-5 \
  --output_dir checkpoints \
  --use_loqt true \
  --single_gpu \
  --with_tracking \
  --report_to wandb 
