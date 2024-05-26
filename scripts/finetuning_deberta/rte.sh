# LoQT rte r=32
python run_glue.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name rte \
  --num_train_epochs 20 \
  --seed 9876 \
  --lora_r 32 \
  --lora_alpha 2 \
  --update_proj_gap 2400 \
  --bnb_4bit_quant_type nf4 \
  --quantize_w '4bit' \
  --quantize_projection_matrix '4bit' \
  --compensate_quant_error_iterations 5 \
  --max_length 256 \
  --pad_to_max_length \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --output_dir checkpoints \
  --use_loqt true \
  --single_gpu \
  --with_tracking \
  --report_to wandb 

# LoQT-nq rte r=32
python run_glue.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name rte \
  --num_train_epochs 20 \
  --seed 9876 \
  --lora_r 32 \
  --lora_alpha 2 \
  --update_proj_gap 2400 \
  --max_length 256 \
  --pad_to_max_length \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --output_dir checkpoints \
  --use_loqt true \
  --single_gpu \
  --with_tracking \
  --report_to wandb 
