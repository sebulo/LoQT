# --nproc_per_node 1 is number of GPUs per node
torchrun --standalone --nproc_per_node 1 --nnodes 1 torchrun_main_hook.py \
    --model_config configs/llama_60m.json \
    --seed 42 \
    --lr 0.01 \
    --rank 128 \
    --lora_alpha 0.4 \
    --update_proj_gap 100 \
    --batch_size 512 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --eval_every 1000 \
    --save_every 1000 \
    --dtype bfloat16 \
    --optimizer adamw \
    --use_loqt True\
    --proj_gap_progression "exponential" \
    --increment_size 1.2 \
    --save_original_model True \
    --only_train_lora True \
    --name 60m_LoQT
    #--quantize_w '4bit' \
    #--quantize_projection_matrix '4bit' \
    #--compensate_quant_error_iterations 5 \
