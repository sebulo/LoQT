 
# loqt - our code: run on h200
# galore - our code: optimizer to galore, use_loqt false
# full - our code: adam, h200

# lora - new code


torchrun --standalone --nproc_per_node 4 torchrun_main.py \
    --model_name /iopsstor/scratch/cscs/vsnbjarn/Llama-2-7b-hf \
    --use_hf_model \
    --lr 0.0002 \
    --galore_scale 0.25 \
    --rank 1024 \
    --update_proj_gap 500 \
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
    --save_dir checkpoints/llama2_7b_c4 \
    --proj_gap_progression "exponential" \
    --increment_size 1.2 \
    --name llam2_7b_1.2_500_c4_loqt_2 \
    --use_offloading True \
    --use_loqt True \
    --wandb_entity PLoRAQ \
    --wandb_project rebuttal