# --nproc_per_node 1 is number of GPUs per node
torchrun --standalone --nproc_per_node 1 --nnodes 1 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --seed 42 \
    --lr 0.01 \
    --rank 128 \
    --lora_alpha 0.4 \
    --update_proj_gap 50 \
    --batch_size 512 \
    --total_batch_size 512 \
    --num_training_steps 500 \
    --warmup_steps 10 \
    --eval_every 0 \
    --save_every 100 \
    --dtype bfloat16 \
    --optimizer adamw \
    --use_loqt True\
    --quantize_w '4bit' \
    --quantize_projection_matrix '4bit' \
    --compensate_quant_error_iterations 5 \
    --proj_gap_progression "exponential" \
    --increment_size 1.2 \
    --single_gpu \
    --run_final_eval False \
    --name 60m_LoQT
    #--continue_from 'checkpoints/60m_LoQT_1716997317/latest_checkpoint' \
