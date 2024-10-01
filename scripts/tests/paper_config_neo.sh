# #!/bin/bash

# #BSUB -q p1                          # Specify queue
# #BSUB -J simple_gpu_job              # Set the job name
# #BSUB -n 16                          # Request number of cores (default: 1)
# #BSUB -R "span[hosts=1]"             # Specify cores must be on the same host
# #BSUB -R "rusage[mem=32GB]"           # Specify 4GB of memory per core/slot
# #BSUB -W 72:00                       # Set walltime limit: hh:mm#BSUB -B
# #BSUB -o output_files/job.%J.out     # Specify the output file. %J is the job-id
# #BSUB -e output_files/job.%J.err     # Specify the error file. %J is the job-id

# # Requesting GPU resources
# #BSUB -gpu "num=1:j_exclusive=yes"   # Request 1 GPU, with exclusive access


# echo "Running on $(hostname):"
# nvidia-smi

# eval "$(conda shell.bash hook)"
# conda activate loqt

#put hf token here?
#$HF_TOKEN=....
export CUDA_VISIBLE_DEVICES=1
python run_gsmk.py \
  --model_name_or_path EleutherAI/gpt-neo-1.3B \
  --num_train_epochs 6 \
  --seed 42 \
  --lora_r 64 \
  --lora_alpha 2 \
  --train_all_params False \
  --num_warmup_steps_procentage 0.03 \
  --update_proj_gap 100000 \
  --max_length 512 \
  --bnb_4bit_quant_type nf4 \
  --quantize_w '4bit' \
  --quantize_projection_matrix '4bit' \
  --pad_to_max_length \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --output_dir checkpoints \
  --use_loqt true \
  --use_offloading True\
  --single_gpu \
  --with_tracking \
  --report_to wandb \
  --hub_token $HF_TOKEN \
  --experiment_name gsm8k_7b_4bit_quant

