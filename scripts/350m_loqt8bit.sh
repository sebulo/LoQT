# Description: Script to run LoQT with 8-bit quantization on 350m model
# nproc_per_node 1 is number of GPUs per node and nnodes 1 is number of nodes
torchrun --standalone --nproc_per_node 1 --nnodes 1 torchrun_main.py \
    --model_config configs/llama_350m.json \
    --seed 42 \
    --lr 0.01 \
    --rank 256 \
    --lora_alpha 0.5 \
    --update_proj_gap 100 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --eval_every 2000 \
    --save_every 10000 \
    --dtype bfloat16 \
    --use_loqt True\
    --quantize_w '4bit' \
    --quantize_projection_matrix '4bit' \
    --compensate_quant_error_iterations 5 \
    --proj_gap_progression "exponential" \
    --increment_size 1.2 \
    --single_gpu \
    --optimizer adam8bit \
    --name 350m_LoQT_8bit
    