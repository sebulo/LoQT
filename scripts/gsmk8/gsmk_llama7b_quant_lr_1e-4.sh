#!/bin/bash
python run_gsmk.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --num_train_epochs 6 \
  --seed 11 \
  --lora_r 64 \
  --lora_alpha 2 \
  --update_proj_gap 100000 \
  --max_length 512 \
  --pad_to_max_length \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --output_dir checkpoints \
  --use_loqt true \
  --single_gpu \
  --compensate_quant_error_iterations 5 \
  --bnb_4bit_quant_type nf4 \
  --quantize_w '4bit' \
  --quantize_projection_matrix '4bit' \
  --with_tracking \
  --report_to wandb 
