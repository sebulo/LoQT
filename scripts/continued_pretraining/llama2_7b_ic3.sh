 
# loqt - our code: run on h200
# galore - our code: optimizer to galore, use_loqt false
# full - our code: adam, h200

# lora - new code


torchrun --standalone --nproc_per_node 4 torchrun_main.py \
    --model_name /iopsstor/scratch/cscs/vsnbjarn/Llama-2-7b-hf \
    --dataset_name /iopsstor/scratch/cscs/vsnbjarn/raw_pretraining_data \
    --use_hf_model \
    --lr 0.0002 \
    --galore_scale 0.25 \
    --rank 1024 \
    --update_proj_gap 10000 \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_training_steps 1500 \
    --warmup_steps 150 \
    --dtype bfloat16 \
    --eval_every 300 \
    --optimizer adam8bit \
    --lora_alpha 0.25 \
    --bnb_4bit_quant_type nf4 \
    --quantize_w '4bit' \
    --quantize_projection_matrix '4bit' \
    --seed 42 \
    --save_every 2000 \
    --save_dir checkpoints/llama2_7b_ice \
    --proj_gap_progression "exponential" \
    --increment_size 1.2 \
    --name llam2_7b_1.2_10k_is_loqt \
    --use_offloading True \
    --use_loqt True \
    --wandb_entity PLoRAQ \
    --wandb_project rebuttal