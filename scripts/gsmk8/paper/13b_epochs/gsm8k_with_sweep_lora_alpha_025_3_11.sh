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

#put hf token here?
HF_TOKEN=$(cat /zhome/0d/2/211877/.cache/huggingface/token)
# Export the variable so that it is accessible in the current shell session
export HF_TOKEN

# Optional: Print to verify
echo "Hugging Face token has been set as: $HF_TOKEN"


  # python run_gsmk.py \
  # --model_name_or_path meta-llama/Llama-2-13b-hf \
  # --num_train_epochs 3 \
  # --seed 11 \
  # --lora_r 64 \
  # --lora_alpha 0.25 \
  # --train_all_params False \
  # --num_warmup_steps_procentage 0.03 \
  # --update_proj_gap 100000 \
  # --max_length 512 \
  # --bnb_4bit_quant_type nf4 \
  # --quantize_w '4bit' \
  # --quantize_projection_matrix '4bit' \
  # --pad_to_max_length \
  # --per_device_train_batch_size 8 \
  # --gradient_accumulation_steps 1 \
  # --learning_rate 3e-4 \
  # --output_dir checkpoints \
  # --use_loqt true \
  # --use_offloading True\
  # --single_gpu \
  # --with_tracking \
  # --report_to wandb \
  # --hub_token $HF_TOKEN \
  # --experiment_name gsm8k_13b_4bit_quant
  
python run_gsmk.py \
  --model_name_or_path meta-llama/Llama-2-13b-hf \
  --num_train_epochs 3 \
  --seed 11 \
  --lora_r 64 \
  --lora_alpha 0.25 \
  --train_all_params False \
  --num_warmup_steps_procentage 0.03 \
  --update_proj_gap 100000 \
  --max_length 512 \
  --bnb_4bit_quant_type nf4 \
  --quantize_w '4bit' \
  --quantize_projection_matrix '4bit' \
  --pad_to_max_length \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 4e-4 \
  --output_dir checkpoints \
  --use_loqt true \
  --use_offloading True\
  --single_gpu \
  --with_tracking \
  --report_to wandb \
  --hub_token $HF_TOKEN \
  --experiment_name gsm8k_13b_4bit_quant

python run_gsmk.py \
  --model_name_or_path meta-llama/Llama-2-13b-hf \
  --num_train_epochs 3 \
  --seed 22 \
  --lora_r 64 \
  --lora_alpha 0.25 \
  --train_all_params False \
  --num_warmup_steps_procentage 0.03 \
  --update_proj_gap 100000 \
  --max_length 512 \
  --bnb_4bit_quant_type nf4 \
  --quantize_w '4bit' \
  --quantize_projection_matrix '4bit' \
  --pad_to_max_length \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 4e-4 \
  --output_dir checkpoints \
  --use_loqt true \
  --use_offloading True\
  --single_gpu \
  --with_tracking \
  --report_to wandb \
  --hub_token $HF_TOKEN \
  --experiment_name gsm8k_13b_4bit_quant

python run_gsmk.py \
  --model_name_or_path meta-llama/Llama-2-13b-hf \
  --num_train_epochs 3 \
  --seed 42 \
  --lora_r 64 \
  --lora_alpha 0.25 \
  --train_all_params False \
  --num_warmup_steps_procentage 0.03 \
  --update_proj_gap 100000 \
  --max_length 512 \
  --bnb_4bit_quant_type nf4 \
  --quantize_w '4bit' \
  --quantize_projection_matrix '4bit' \
  --pad_to_max_length \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 4e-4 \
  --output_dir checkpoints \
  --use_loqt true \
  --use_offloading True\
  --single_gpu \
  --with_tracking \
  --report_to wandb \
  --hub_token $HF_TOKEN \
  --experiment_name gsm8k_13b_4bit_quant


python run_gsmk.py \
  --model_name_or_path meta-llama/Llama-2-13b-hf \
  --num_train_epochs 3 \
  --seed 11 \
  --lora_r 64 \
  --lora_alpha 0.25 \
  --train_all_params False \
  --num_warmup_steps_procentage 0.03 \
  --update_proj_gap 100000 \
  --max_length 512 \
  --bnb_4bit_quant_type nf4 \
  --quantize_w '4bit' \
  --quantize_projection_matrix '4bit' \
  --pad_to_max_length \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --output_dir checkpoints \
  --use_loqt true \
  --use_offloading True\
  --single_gpu \
  --with_tracking \
  --report_to wandb \
  --hub_token $HF_TOKEN \
  --experiment_name gsm8k_13b_4bit_quant


  python run_gsmk.py \
  --model_name_or_path meta-llama/Llama-2-13b-hf \
  --num_train_epochs 3 \
  --seed 11 \
  --lora_r 64 \
  --lora_alpha 0.25 \
  --train_all_params False \
  --num_warmup_steps_procentage 0.03 \
  --update_proj_gap 100000 \
  --max_length 512 \
  --bnb_4bit_quant_type nf4 \
  --quantize_w '4bit' \
  --quantize_projection_matrix '4bit' \
  --pad_to_max_length \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --output_dir checkpoints \
  --use_loqt true \
  --use_offloading True\
  --single_gpu \
  --with_tracking \
  --report_to wandb \
  --hub_token $HF_TOKEN \
  --experiment_name gsm8k_13b_4bit_quant
  
python run_gsmk.py \
  --model_name_or_path meta-llama/Llama-2-13b-hf \
  --num_train_epochs 3 \
  --seed 11 \
  --lora_r 64 \
  --lora_alpha 0.25 \
  --train_all_params False \
  --num_warmup_steps_procentage 0.03 \
  --update_proj_gap 100000 \
  --max_length 512 \
  --bnb_4bit_quant_type nf4 \
  --quantize_w '4bit' \
  --quantize_projection_matrix '4bit' \
  --pad_to_max_length \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 7e-5 \
  --output_dir checkpoints \
  --use_loqt true \
  --use_offloading True\
  --single_gpu \
  --with_tracking \
  --report_to wandb \
  --hub_token $HF_TOKEN \
  --experiment_name gsm8k_13b_4bit_quant

