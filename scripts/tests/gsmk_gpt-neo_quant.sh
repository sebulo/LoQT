#!/bin/bash
python run_gsmk.py \
  --model_name_or_path EleutherAI/gpt-neo-1.3B \
  --experiment_name gsmk_gpt-neo-1.3B_seed9876 \
  --compensate_quant_error_iterations 5 \
  --bnb_4bit_quant_type nf4 \
  --quantize_projection_matrix '4bit' \
  --quantize_w '4bit' \
  --num_train_epochs 6 \
  --seed 9876 \
  --lora_r 32 \
  --lora_alpha 2 \
  --update_proj_gap 1000 \
  --max_length 256 \
  --pad_to_max_length \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 3e-4 \
  --output_dir checkpoints/gpt-neo-1.3B \
  --use_loqt true \
  --single_gpu \
  --with_tracking \
  --report_to wandb 
  #--max_train_steps 1000 \
#--warmup_ratio 0.03 \
#--model_name_or_path LoftQ/Llama-2-7b-hf-4bit-64rank \
#--model_name_or_path checkpoints/llama7b \

# LoftQ: train 4-bit 64-rank llama-2-7b with LoftQ on GSM8K using 8 A100s
# global batch_size=64
# accelerate launch train_gsm8k.py \
#   --model_name_or_path LoftQ/Llama-2-7b-hf-4bit-64rank \
#   --learning_rate 3e-4 \
#   --seed 11 \
#   --expt_name gsm8k_llama2_7b_4bit_64rank_loftq_fake \
#   --output_dir exp_results/ \
#   --num_train_epochs 6 \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 4 \
#   --evaluation_strategy "no" \
#   --save_strategy "epoch" \
#   --weight_decay 0.1 \
#   --warmup_ratio 0.03 \
#   --lr_scheduler_type "cosine" \
#   --logging_steps 10 \
#   --do_train \
#   --report_to tensorboard