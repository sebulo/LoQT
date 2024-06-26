# --nproc_per_node 4 is number of GPUs per node
torchrun --standalone --nproc_per_node 4 --nnodes 1 torchrun_main.py \
    --model_config configs/llama_350m.json \
    --seed 42 \
    --lr 0.01 \
    --rank 256 \
    --lora_alpha 0.25 \
    --update_proj_gap 100 \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_training_steps 100000 \
    --warmup_steps 10000 \
    --eval_every 2000 \
    --save_every 10000 \
    --dtype bfloat16 \
    --optimizer adamw \
    --use_loqt True\
    --bnb_4bit_quant_type nf4 \
    --quantize_w '4bit' \
    --quantize_projection_matrix '4bit' \
    --compensate_quant_error_iterations 10 \
    --proj_gap_progression "exponential" \
    --increment_size 1.2 \
    --name 1b_LoQT
