export CUDA_VISIBLE_DEVICES=0

#!/bin/bash

# Function to perform cleanup
cleanup() {
    echo "Caught signal, cleaning up..."
    exit 1  # Exit the script with an error status
}

timeout_n=3600
# Trap SIGINT and SIGTERM
trap cleanup SIGINT SIGTERM

# Define a flag file for controlling execution
flag_file="flag_file"

# Create the flag file to start
touch "$flag_file"
COMMON_ARGS="--lr 0.01 --galore_scale 0.25 --lora_alpha 0.5 --proj_type std --update_proj_gap 2 --batch_size 1 --total_batch_size 2 --num_training_steps 3 --scheduler_effective_training_steps 10000 --warmup_steps 1000 --weight_decay 0 --dtype bfloat16 --eval_every 500 --save_every 0 --num_eval_tokens 1000 --single_gpu --log_max_memory true --run_final_eval false"


# Define model configurations with their corresponding ranks
declare -a models_and_ranks=("llama_13b:1024")


# Run specific adamW configs with specific ranks
for model_and_rank in "${models_and_ranks[@]}"; do
    model="${model_and_rank%%:*}"  # Extract model name before the colon
    rank="${model_and_rank##*:}"   # Extract rank after the colon

    if [[ -f "$flag_file" ]]; then
        timeout ${timeout_n} torchrun --standalone --nproc_per_node 1 --nnodes 1 torchrun_main.py \
            --model_config "configs/${model}.json" \
            $COMMON_ARGS \
            --rank ${rank} \
            --optimizer adamw8bit_per_layer \
            --use_loqt True \
            --compensate_quant_error_iterations 0 \
            --use_offloading True \
            --bnb_4bit_quant_type nf4 \
            --quantize_w '4bit' \
            --use_double_quant True \
            --quantize_projection_matrix '4bit' \
            --use_eigenh_for_projection True\
            --name "memprof_loqt8bit_layerwise_${model}_adam8bit_offload" || true
    else
        echo "Flag file does not exist, stopping execution."
        break
    fi
done


# Cleanup at the end
rm -f "${flag_file}"
echo "Script completed."